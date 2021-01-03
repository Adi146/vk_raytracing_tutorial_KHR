#include "gbuffer.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvmath/nvmath_glsltypes.h"

void GBuffer::destroy()
{
  Renderpass::destroy();
  m_device.destroy(m_RenderPass);
  m_device.destroy(m_Framebuffer);

  m_alloc->destroy(m_position);
  m_alloc->destroy(m_normal);
  m_alloc->destroy(m_color);
  m_alloc->destroy(m_depth);
}

void GBuffer::createRender(vk::Extent2D size)
{
  Renderpass::createRender(size);

  m_alloc->destroy(m_position);
  m_alloc->destroy(m_normal);
  m_alloc->destroy(m_color);

  //position map
  {
    auto positionCreateInfo = nvvk::makeImage2DCreateInfo(size, m_positionColorFormat,
                                                          vk::ImageUsageFlagBits::eColorAttachment |
                                                          vk::ImageUsageFlagBits::eSampled | 
                                                          vk::ImageUsageFlagBits::eStorage);
    nvvk::Image image = m_alloc->createImage(positionCreateInfo);
    vk::ImageViewCreateInfo ivInfo;
    ivInfo.setViewType(vk::ImageViewType::e2D);
    ivInfo.setFormat(m_positionColorFormat);
    ivInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    ivInfo.setImage(image.image);

    m_position = m_alloc->createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_position.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  //normal map
  {
    auto normalCreateInfo = nvvk::makeImage2DCreateInfo(size, m_normalColorFormat,
                                                        vk::ImageUsageFlagBits::eColorAttachment |
                                                        vk::ImageUsageFlagBits::eSampled |
                                                        vk::ImageUsageFlagBits::eStorage);
    nvvk::Image image = m_alloc->createImage(normalCreateInfo);
    vk::ImageViewCreateInfo ivInfo;
    ivInfo.setViewType(vk::ImageViewType::e2D);
    ivInfo.setFormat(m_normalColorFormat);
    ivInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    ivInfo.setImage(image.image);

    m_normal = m_alloc->createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_normal.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  //color map
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(size, m_colorColorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment |
                                                       vk::ImageUsageFlagBits::eSampled |
                                                       vk::ImageUsageFlagBits::eStorage);
    nvvk::Image image = m_alloc->createImage(colorCreateInfo);
    vk::ImageViewCreateInfo ivInfo;
    ivInfo.setViewType(vk::ImageViewType::e2D);
    ivInfo.setFormat(m_colorColorFormat);
    ivInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
    ivInfo.setImage(image.image);

    m_color = m_alloc->createTexture(image, ivInfo, vk::SamplerCreateInfo());
    m_color.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  //depth map
  {
    auto depthCreateInfo = nvvk::makeImage2DCreateInfo(size, m_depthColorFormat,
                                                       vk::ImageUsageFlagBits::eDepthStencilAttachment);
    nvvk::Image image = m_alloc->createImage(depthCreateInfo);
    vk::ImageViewCreateInfo ivInfo;
    ivInfo.setViewType(vk::ImageViewType::e2D);
    ivInfo.setFormat(m_depthColorFormat);
    ivInfo.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
    ivInfo.setImage(image.image);

    m_depth = m_alloc->createTexture(image, ivInfo);
  }

  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_position.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_normal.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_color.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_depth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the g-buffer
  if(!m_RenderPass)
  {
    m_RenderPass =
        nvvk::createRenderPass(m_device,
                               {m_positionColorFormat, m_normalColorFormat, m_colorColorFormat},
                               m_depthColorFormat, 1, true, true, vk::ImageLayout::eGeneral,
                               vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for g-buffer
  std::vector<vk::ImageView> attachments = {m_position.descriptor.imageView,
                                            m_normal.descriptor.imageView,
                                            m_color.descriptor.imageView,
                                            m_depth.descriptor.imageView};

  m_device.destroy(m_Framebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_RenderPass);
  info.setAttachmentCount(4);
  info.setPAttachments(attachments.data());
  info.setWidth(size.width);
  info.setHeight(size.height);
  info.setLayers(1);
  m_Framebuffer = m_device.createFramebuffer(info);
}

void GBuffer::createPipeline
(
  vk::DescriptorSetLayout* descSetLayout,
  std::vector<std::string> defaultSearchPaths
)
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eVertex
                                                  | vk::ShaderStageFlagBits::eFragment,
                                              0, sizeof(PushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_PipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_PipelineLayout,
                                                            m_RenderPass);
  vk::PipelineColorBlendAttachmentState   blendState;
  blendState.blendEnable    = VK_FALSE;
  blendState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                              | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

  pipelineGenerator.setBlendAttachmentState(0, blendState);
  pipelineGenerator.addBlendAttachmentState(blendState);
  pipelineGenerator.addBlendAttachmentState(blendState);
  pipelineGenerator.addShader(nvh::loadFile("shaders/gBuffer.vert.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/gBuffer.frag.spv", true, paths, true),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.addBindingDescriptions(
      {{0, sizeof(nvmath::vec3)}, {1, sizeof(nvmath::vec3)}, {2, sizeof(nvmath::vec2)}});
  pipelineGenerator.addAttributeDescriptions({
      {0, 0, vk::Format::eR32G32B32Sfloat, 0},  // Position
      {1, 1, vk::Format::eR32G32B32Sfloat, 0},  // Normal
      {2, 2, vk::Format::eR32G32Sfloat, 0},     // Texcoord0
  });
  m_Pipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_Pipeline, "gBuffer");
}

void GBuffer::draw
(
  const vk::CommandBuffer& cmdBuf,
  vk::DescriptorSet        descSet,
  std::vector<vk::Buffer>  vertexBuffers,
  vk::Buffer               indexBuffer,
  nvh::GltfScene&          gltfScene
)
{
  vk::ClearValue clearValues[4];
  clearValues[0].setColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f});
  clearValues[1].setColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f});
  clearValues[2].setColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f});
  clearValues[3].setDepthStencil({ 1.0f, 0 });

  vk::RenderPassBeginInfo renderPassBeginInfo;
  renderPassBeginInfo.setClearValueCount(4);
  renderPassBeginInfo.setPClearValues(clearValues);
  renderPassBeginInfo.setRenderPass(m_RenderPass);
  renderPassBeginInfo.setFramebuffer(m_Framebuffer);
  renderPassBeginInfo.setRenderArea({ {}, m_outputSize });

  cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

  std::vector<vk::DeviceSize> offsets = {0, 0, 0};
  m_debug.beginLabel(cmdBuf, "GBuffer");

  // Dynamic Viewport
  cmdBuf.setViewport
  (
    0, 
    {
      vk::Viewport(0, 0, (float)m_outputSize.width, (float)m_outputSize.height, 0, 1)
    }
  );
  cmdBuf.setScissor(0, {{{0, 0}, {m_outputSize.width, m_outputSize.height}}});

  // Drawing all triangles
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_Pipeline);
  cmdBuf.bindDescriptorSets
  (
    vk::PipelineBindPoint::eGraphics,
    m_PipelineLayout, 
    0,
    {descSet}, 
    {}
  );
  cmdBuf.bindVertexBuffers
  (
    0,
    static_cast<uint32_t>(vertexBuffers.size()),
    vertexBuffers.data(),
    offsets.data()
  );
  cmdBuf.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);

  uint32_t idxNode = 0;
  for(auto& node : gltfScene.m_nodes)
  {
    auto& primitive = gltfScene.m_primMeshes[node.primMesh];

    m_pushConstant.instanceId = idxNode++;
    m_pushConstant.materialId = primitive.materialIndex;
    cmdBuf.pushConstants<PushConstant>
    (
      m_PipelineLayout,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      0,
      m_pushConstant
    );
    cmdBuf.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);
  }

  m_debug.endLabel(cmdBuf);

  cmdBuf.endRenderPass();
}