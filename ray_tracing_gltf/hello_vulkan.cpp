/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <sstream>
#include <vulkan/vulkan.hpp>

extern std::vector<std::string> defaultSearchPaths;

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/gltfscene.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "nvh/alignment.hpp"
#include "shaders/binding.glsl"
#include "shaders/gltf.glsl"

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Instance&       instance,
                        const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);

  m_gbuffer.setup(device, physicalDevice, queueFamily, &m_alloc);
  m_postprocessing.setup(device, physicalDevice, queueFamily, &m_alloc);
  m_pathtrace.setup(device, physicalDevice, queueFamily, &m_alloc);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const vk::CommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  CameraMatrices hostUBO = {};
  hostUBO.view           = CameraManip.getMatrix();
  hostUBO.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // hostUBO.proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).
  hostUBO.viewInverse = nvmath::invert(hostUBO.view);
  // #VKRay
  hostUBO.projInverse = nvmath::invert(hostUBO.proj);

  // UBO on the device, and what stages access it.
  vk::Buffer deviceUBO = m_cameraMat.buffer;
  auto uboUsageStages = vk::PipelineStageFlagBits::eVertexShader
                      | vk::PipelineStageFlagBits::eRayTracingShaderKHR;

  // Ensure that the modified UBO is not visible to previous frames.
  vk::BufferMemoryBarrier beforeBarrier;
  beforeBarrier.setSrcAccessMask(vk::AccessFlagBits::eShaderRead);
  beforeBarrier.setDstAccessMask(vk::AccessFlagBits::eTransferWrite);
  beforeBarrier.setBuffer(deviceUBO);
  beforeBarrier.setOffset(0);
  beforeBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(
    uboUsageStages,
    vk::PipelineStageFlagBits::eTransfer,
    vk::DependencyFlagBits::eDeviceGroup, {}, {beforeBarrier}, {});

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  cmdBuf.updateBuffer<CameraMatrices>(m_cameraMat.buffer, 0, hostUBO);

  // Making sure the updated UBO will be visible.
  vk::BufferMemoryBarrier afterBarrier;
  afterBarrier.setSrcAccessMask(vk::AccessFlagBits::eTransferWrite);
  afterBarrier.setDstAccessMask(vk::AccessFlagBits::eShaderRead);
  afterBarrier.setBuffer(deviceUBO);
  afterBarrier.setOffset(0);
  afterBarrier.setSize(sizeof hostUBO);
  cmdBuf.pipelineBarrier(
    vk::PipelineStageFlagBits::eTransfer,
    uboUsageStages,
    vk::DependencyFlagBits::eDeviceGroup, {}, {afterBarrier}, {});
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());

  auto& bind = m_descSetLayoutBind;
  // Camera matrices (binding = 0)
  bind.addBinding(vkDS(B_CAMERA, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenKHR));
  bind.addBinding(
      vkDS(B_VERTICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(
      vkDS(B_INDICES, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_NORMALS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_TEXCOORDS, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));
  bind.addBinding(vkDS(B_MATERIALS, vkDT::eStorageBuffer, 1,
                       vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  bind.addBinding(vkDS(B_MATRICES, vkDT::eStorageBuffer, 1,
                       vkSS::eVertex | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(vkDS(B_TEXTURES, vkDT::eCombinedImageSampler, nbTextures,
                       vkSS::eFragment | vkSS::eClosestHitKHR | vkSS::eAnyHitKHR));


  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo uvDesc{m_uvBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo matrixDesc{m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE};

  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_CAMERA, &dbiUnif));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_VERTICES, &vertexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_INDICES, &indexDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_NORMALS, &normalDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_TEXCOORDS, &uvDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATERIALS, &materialDesc));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, B_MATRICES, &matrixDesc));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, B_TEXTURES, diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadScene(const std::string& filename)
{
  using vkBU = vk::BufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("Loading file: %s", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    assert(!"Error while loading scene");
  }
  LOGW(warn.c_str());
  LOGE(error.c_str());


  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel,
                                  nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = cmdBufGet.createCommandBuffer();

  m_vertexBuffer =
      m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
                           vkBU::eVertexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  m_indexBuffer =
      m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
                           vkBU::eIndexBuffer | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress
                               | vkBU::eAccelerationStructureBuildInputReadOnlyKHR);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
                                        vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0,
                                    vkBU::eVertexBuffer | vkBU::eStorageBuffer);

  // Copying all materials, only the elements we need
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
    shadeMaterials.emplace_back(
        GltfShadeMaterial{m.pbrBaseColorFactor, m.pbrBaseColorTexture, m.emissiveFactor});
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, vkBU::eStorageBuffer);

  // Instance Matrices used by rasterizer
  std::vector<nvmath::mat4f> nodeMatrices;
  for(auto& node : m_gltfScene.m_nodes)
  {
    nodeMatrices.emplace_back(node.worldMatrix);
  }
  m_matrixBuffer = m_alloc.createBuffer(cmdBuf, nodeMatrices, vkBU::eStorageBuffer);

  // The following is used to find the primitive mesh information in the CHIT
  std::vector<RtPrimitiveLookup> primLookup;
  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    primLookup.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
  }
  m_rtPrimLookup =
      m_alloc.createBuffer(cmdBuf, primLookup, vk::BufferUsageFlagBits::eStorageBuffer);


  // Creates all textures found
  createTextureImages(cmdBuf, tmodel);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_uvBuffer.buffer, "TexCoord");
  m_debug.setObjectName(m_materialBuffer.buffer, "Material");
  m_debug.setObjectName(m_matrixBuffer.buffer, "Matrix");
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices),
                                     vkBU::eUniformBuffer | vkBU::eTransferDst, vkMP::eDeviceLocal);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  auto addDefaultTexture = [this]() {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_alloc.createTexture(
        cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto&        gltfimage  = gltfModel.images[i];
    void*        buffer     = &gltfimage.image[0];
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = vk::Extent2D(gltfimage.width, gltfimage.height);

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultTexture();
      continue;
    }

    vk::ImageCreateInfo imageCreateInfo =
        nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures.emplace_back(m_alloc.createTexture(image, ivInfo, samplerCreateInfo));

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);

  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_matrixBuffer);
  m_alloc.destroy(m_rtPrimLookup);

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_postprocessing.destroy();

  //# GBuffer
  m_gbuffer.destroy();

  //#A-Trous
  m_device.destroy(m_aTrousPipeline);
  m_device.destroy(m_aTrousPipelineLayout);
  m_device.destroy(m_aTrousDescPool);
  m_device.destroy(m_aTrousDescSetLayout);
  m_alloc.destroy(m_aTrousTexturePing);
  m_alloc.destroy(m_aTrousTexturePong);
  m_device.destroy(m_aTrousRenderPass);
  m_device.destroy(m_aTrousFramebufferPing);
  m_device.destroy(m_aTrousFramebufferPong);

  // #VKRay
  m_pathtrace.destroy();
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  m_pathtrace.createRender(m_size);
  m_postprocessing.createRender(m_size, getRenderPass());
  m_gbuffer.createRender(m_size);
  createATrousRender();

  updatePostDescriptorSet();
  updateATrousDescriptorSet();
  updateRtDescriptorSet();
}

void HelloVulkan::createATrousRender()
{
  m_alloc.destroy(m_aTrousTexturePing);
  m_alloc.destroy(m_aTrousTexturePong);

  {
    auto textureCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_aTrousFormat,
                                                          vk::ImageUsageFlagBits::eColorAttachment |
                                                          vk::ImageUsageFlagBits::eSampled |
                                                          vk::ImageUsageFlagBits::eStorage);
    nvvk::Image ping       = m_alloc.createImage(textureCreateInfo);
    nvvk::Image pong       = m_alloc.createImage(textureCreateInfo);

    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(ping.image, textureCreateInfo);
    m_aTrousTexturePing = m_alloc.createTexture(ping, ivInfo, vk::SamplerCreateInfo());

    ivInfo.setImage(pong.image);
    m_aTrousTexturePong = m_alloc.createTexture(pong, ivInfo, vk::SamplerCreateInfo());

    m_aTrousTexturePing.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_aTrousTexturePong.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_aTrousTexturePing.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_aTrousTexturePong.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    genCmdBuf.submitAndWait(cmdBuf);
  }

  if(!m_aTrousRenderPass)
  {
    m_aTrousRenderPass =
        nvvk::createRenderPass(m_device, { m_aTrousFormat }, vk::Format::eUndefined, 1, true,
                               true, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
  }


  m_device.destroy(m_aTrousFramebufferPing);
  m_device.destroy(m_aTrousFramebufferPong);

  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_aTrousRenderPass);
  info.setAttachmentCount(1);
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);

  std::vector<vk::ImageView> attachments = {m_aTrousTexturePing.descriptor.imageView};
  info.setPAttachments(attachments.data());
  m_aTrousFramebufferPing = m_device.createFramebuffer(info);

  attachments = {m_aTrousTexturePong.descriptor.imageView};
  info.setPAttachments(attachments.data());
  m_aTrousFramebufferPong = m_device.createFramebuffer(info);
}

void HelloVulkan::createATrousPipeline()
{
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(ATrousPushConstants)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_aTrousDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_aTrousPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_aTrousPipelineLayout,
                                                            m_aTrousRenderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/a-trous.frag.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_aTrousPipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_aTrousPipeline, "a-trous");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createATrousDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_aTrousDescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment | vkSS::eVertex));
  m_aTrousDescSetLayoutBind.addBinding(vkDS(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_aTrousDescSetLayoutBind.addBinding(vkDS(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_aTrousDescSetLayout = m_aTrousDescSetLayoutBind.createLayout(m_device);
  m_aTrousDescPool      = m_aTrousDescSetLayoutBind.createPool(m_device, 2);

  m_aTrousDescSetPing   = nvvk::allocateDescriptorSet(m_device, m_aTrousDescPool, m_aTrousDescSetLayout);
  m_aTrousDescSetPong   = nvvk::allocateDescriptorSet(m_device, m_aTrousDescPool, m_aTrousDescSetLayout);
}

void HelloVulkan::updateATrousDescriptorSet()
{
  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPing, 0, &m_aTrousTexturePong.descriptor));
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPing, 1, &m_gbuffer.m_position.descriptor));
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPing, 2, &m_gbuffer.m_normal.descriptor));

    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPong, 0, &m_aTrousTexturePing.descriptor));
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPong, 1, &m_gbuffer.m_position.descriptor));
    writes.emplace_back(m_aTrousDescSetLayoutBind.makeWrite(m_aTrousDescSetPong, 2, &m_gbuffer.m_normal.descriptor));

    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }
}

void HelloVulkan::drawATrous(const vk::CommandBuffer& cmdBuf)
{
  if (!m_enableATrous)
    return;

  vk::ClearValue clearValues[1];
  clearValues[0].setColor(std::array<float, 4>({0, 0, 0, 0}));

  m_debug.beginLabel(cmdBuf, "A-Trous");


  for (int i = 0; i <= 5; i++)
  {
    m_aTrousPushConstant.stepwidth = i * 2 + 1;
    m_aTrousPushConstant.c_phi     = 1.0f / i * m_c_phi0;
    m_aTrousPushConstant.n_phi     = 1.0f / i * m_n_phi0;
    m_aTrousPushConstant.p_phi     = 1.0f / i * m_p_phi0;

    vk::RenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.setClearValueCount(1);
    renderPassBeginInfo.setPClearValues(clearValues);
    renderPassBeginInfo.setRenderPass(m_aTrousRenderPass);
    renderPassBeginInfo.setFramebuffer((i % 2 == 0) ? m_aTrousFramebufferPong :
                                                      m_aTrousFramebufferPing);
    renderPassBeginInfo.setRenderArea({{}, m_size});

    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    // Rendering tonemapper
    cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
    cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

    cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_aTrousPipeline);
    cmdBuf.pushConstants<ATrousPushConstants>(
        m_aTrousPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_aTrousPushConstant);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_aTrousPipelineLayout, 0,
                              (i % 2 == 0) ? m_aTrousDescSetPong : m_aTrousDescSetPing, {});
    cmdBuf.draw(3, 1, 0, 0);

    cmdBuf.endRenderPass();
  }

  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;
  if (m_enableATrous)
  {
    writes.emplace_back(m_postprocessing.m_DescSetLayoutBind.makeWrite
    (
      m_postprocessing.m_DescSet,
      0,
      &m_aTrousTexturePong.descriptor
    ));
  }
  else
  {
    writes.emplace_back(m_postprocessing.m_DescSetLayoutBind.makeWrite
    (
      m_postprocessing.m_DescSet,
      0,
      &m_pathtrace.m_historyColor.descriptor
    ));
  }

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  vk::DescriptorImageInfo outputImageInfo
  {
    {}, m_aTrousTexturePing.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  vk::DescriptorImageInfo historyImageInfo
  {
    {}, m_pathtrace.m_historyColor.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_pathtrace.m_DescSetLayoutBind.makeWrite(m_pathtrace.m_DescSet, 1, &outputImageInfo));
  writes.emplace_back(m_pathtrace.m_DescSetLayoutBind.makeWrite(m_pathtrace.m_DescSet, 2, &historyImageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}



