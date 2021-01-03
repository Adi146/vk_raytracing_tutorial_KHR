#include "pathtrace.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvh/alignment.hpp"

void Pathtrace::setup
(
  const vk::Device&         device,
  const vk::PhysicalDevice& physicalDevice,
  uint32_t                  queueIndex,
  nvvk::Allocator*          allocator
)
{
  Renderpass::setup(device, physicalDevice, queueIndex, allocator);

  // Requesting ray tracing properties
  auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_Properties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_Builder.setup(m_device, allocator, queueIndex);
}

void Pathtrace::destroy()
{
  Renderpass::destroy();

  m_alloc->destroy(m_historyColor);
  m_alloc->destroy(m_outputColor);
  m_alloc->destroy(m_depth);

  m_Builder.destroy();
  m_device.destroy(m_DescPool);
  m_device.destroy(m_DescSetLayout);
  m_alloc->destroy(m_SBTBuffer);
}

void Pathtrace::createRender(vk::Extent2D size)
{
  Renderpass::createRender(size);

  m_alloc->destroy(m_historyColor);
  m_alloc->destroy(m_depth);

  // Creating the color images
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(size, m_colorFormat,
                                                       vk::ImageUsageFlagBits::eColorAttachment |
                                                       vk::ImageUsageFlagBits::eSampled |
                                                       vk::ImageUsageFlagBits::eStorage);
    nvvk::Image             history = m_alloc->createImage(colorCreateInfo);
    nvvk::Image             output = m_alloc->createImage(colorCreateInfo);

    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(history.image, colorCreateInfo);
    m_historyColor = m_alloc->createTexture(history, ivInfo, vk::SamplerCreateInfo());

    ivInfo.setImage(output.image);
    m_outputColor = m_alloc->createTexture(output, ivInfo, vk::SamplerCreateInfo());

    m_historyColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    m_outputColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo =
    nvvk::makeImage2DCreateInfo(size, m_depthFormat,
      vk::ImageUsageFlagBits::eDepthStencilAttachment);
  {
    nvvk::Image image = m_alloc->createImage(depthCreateInfo);

    vk::ImageViewCreateInfo depthStencilView;
    depthStencilView.setViewType(vk::ImageViewType::e2D);
    depthStencilView.setFormat(m_depthFormat);
    depthStencilView.setSubresourceRange({ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 });
    depthStencilView.setImage(image.image);

    m_depth = m_alloc->createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_historyColor.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_outputColor.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eGeneral);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_depth.image, vk::ImageLayout::eUndefined,
                                vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                vk::ImageAspectFlagBits::eDepth);

    genCmdBuf.submitAndWait(cmdBuf);
  }
}

void Pathtrace::createDescriptorSet(vk::Buffer primitiveLookupBuffer)
{
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_DescSetLayoutBind.addBinding // TLAS
  (
    vkDSLB(0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR)
  );
  m_DescSetLayoutBind.addBinding // Output image
  (
    vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR)
  );
  m_DescSetLayoutBind.addBinding // History image
  (
    vkDSLB(2, vkDT::eStorageImage, 1, vkSS::eRaygenKHR)
  );
  m_DescSetLayoutBind.addBinding // Primitive info
  (
    vkDSLB(3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR | vkSS::eAnyHitKHR)
  );

  m_DescPool = m_DescSetLayoutBind.createPool(m_device);
  m_DescSetLayout = m_DescSetLayoutBind.createLayout(m_device);
  m_DescSet = m_device.allocateDescriptorSets({ m_DescPool, 1, &m_DescSetLayout })[0];

  vk::AccelerationStructureKHR                   tlas = m_Builder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureKHR descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&tlas);
  vk::DescriptorImageInfo outputImageInfo
  {
    {}, m_outputColor.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  vk::DescriptorImageInfo historyImageInfo
  {
    {}, m_historyColor.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  vk::DescriptorBufferInfo primitiveInfoDesc{ primitiveLookupBuffer, 0, VK_WHOLE_SIZE };

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 0, &descASInfo));
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 1, &outputImageInfo));
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 2, &historyImageInfo));
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 3, &primitiveInfoDesc));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Pathtrace::createPipeline
(
  vk::DescriptorSetLayout* descSetLayout,
  std::vector<std::string> defaultSearchPaths
)
{
  std::vector<std::string> paths = defaultSearchPaths;

  vk::ShaderModule raygenSM =
    nvvk::createShaderModule(m_device,  //
      nvh::loadFile("shaders/pathtrace.rgen.spv", true, paths, true));
  vk::ShaderModule missSM =
    nvvk::createShaderModule(m_device,  //
      nvh::loadFile("shaders/pathtrace.rmiss.spv", true, paths, true));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM =
    nvvk::createShaderModule(m_device,
      nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths, true));


  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoKHR rg{ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR };
  stages.push_back({ {}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main" });
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_ShaderGroups.push_back(rg);
  // Miss
  vk::RayTracingShaderGroupCreateInfoKHR mg{ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR };
  stages.push_back({ {}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main" });
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_ShaderGroups.push_back(mg);
  // Shadow Miss
  stages.push_back({ {}, vk::ShaderStageFlagBits::eMissKHR, shadowmissSM, "main" });
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_ShaderGroups.push_back(mg);

  // Hit Group - Closest Hit + AnyHit
  vk::ShaderModule chitSM =
    nvvk::createShaderModule(m_device,  //
      nvh::loadFile("shaders/pathtrace.rchit.spv", true, paths, true));

  vk::RayTracingShaderGroupCreateInfoKHR hg{ vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR };
  stages.push_back({ {}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main" });
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_ShaderGroups.push_back(hg);

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{ vk::ShaderStageFlagBits::eRaygenKHR
                                         | vk::ShaderStageFlagBits::eClosestHitKHR
                                         | vk::ShaderStageFlagBits::eMissKHR,
                                     0, sizeof(PushConstant) };
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = { m_DescSetLayout, *descSetLayout };
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_PipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline

  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
    m_ShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_ShaderGroups.data());

  rayPipelineInfo.setMaxPipelineRayRecursionDepth(5);  // Ray depth
  rayPipelineInfo.setLayout(m_PipelineLayout);
  m_Pipeline =
    static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void Pathtrace::createShaderBindingTable()
{
  auto groupCount =
    static_cast<uint32_t>(m_ShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_Properties.shaderGroupHandleSize;  // Size of a program identifier
  uint32_t groupSizeAligned =
    nvh::align_up(groupHandleSize, m_Properties.shaderGroupBaseAlignment);

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupSizeAligned;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  auto result = m_device.getRayTracingShaderGroupHandlesKHR(m_Pipeline, 0, groupCount, sbtSize,
    shaderHandleStorage.data());
  if (result != vk::Result::eSuccess)
    LOGE("Fail getRayTracingShaderGroupHandlesKHR: %s", vk::to_string(result).c_str());

  // Write the handles in the SBT
  m_SBTBuffer = m_alloc->createBuffer(
    sbtSize,
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddressKHR
    | vk::BufferUsageFlagBits::eShaderBindingTableKHR,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_SBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_SBTBuffer);
  auto* pData = reinterpret_cast<uint8_t*>(mapped);
  for (uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += groupSizeAligned;
  }
  m_alloc->unmap(m_SBTBuffer);


  m_alloc->finalizeAndReleaseStaging();
}

void Pathtrace::createBottomLevelAS
(
  const nvh::GltfScene& gltfScene,
  const vk::Buffer vertexBuffer,
  const vk::Buffer indexBuffer
)
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(gltfScene.m_primMeshes.size());
  for (auto& primMesh : gltfScene.m_primMeshes)
  {
    auto geo = primitiveToGeometry(primMesh, vertexBuffer, indexBuffer);
    allBlas.push_back({ geo });
  }
  m_Builder.buildBlas(allBlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

void Pathtrace::createTopLevelAS(const nvh::GltfScene& gltfScene)
{
  std::vector<nvvk::RaytracingBuilderKHR::Instance> tlas;
  tlas.reserve(gltfScene.m_nodes.size());
  uint32_t instID = 0;
  for (auto& node : gltfScene.m_nodes)
  {
    nvvk::RaytracingBuilderKHR::Instance rayInst;
    rayInst.transform = node.worldMatrix;
    rayInst.instanceCustomId = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.blasId = node.primMesh;
    rayInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
    tlas.emplace_back(rayInst);
  }
  m_Builder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput Pathtrace::primitiveToGeometry
(
  const nvh::GltfPrimMesh& prim,
  const vk::Buffer vertexBuffer,
  const vk::Buffer indexBuffer
)
{
  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({ vertexBuffer });
  vk::DeviceAddress indexAddress = m_device.getBufferAddress({ indexBuffer });

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});
  triangles.setMaxVertex(prim.vertexCount);

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);  // For AnyHit
  asGeom.geometry.setTriangles(triangles);

  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(prim.vertexOffset);
  offset.setPrimitiveCount(prim.indexCount / 3);
  offset.setPrimitiveOffset(prim.firstIndex * sizeof(uint32_t));
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

void Pathtrace::updateDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  vk::DescriptorImageInfo outputImageInfo
  {
    {}, m_outputColor.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  vk::DescriptorImageInfo historyImageInfo
  {
    {}, m_historyColor.descriptor.imageView, vk::ImageLayout::eGeneral
  };

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 1, &outputImageInfo));
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 2, &historyImageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void Pathtrace::draw
(
  const vk::CommandBuffer& cmdBuf,
  vk::DescriptorSet        descSet,
  const nvmath::vec4f&     clearColor
)
{
  m_pushConstants.frame++;

  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_pushConstants.clearColor = clearColor;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_Pipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_PipelineLayout, 0,
    { m_DescSet, descSet }, {});
  cmdBuf.pushConstants<PushConstant>(m_PipelineLayout,
    vk::ShaderStageFlagBits::eRaygenKHR
    | vk::ShaderStageFlagBits::eClosestHitKHR
    | vk::ShaderStageFlagBits::eMissKHR,
    0, m_pushConstants);

  // Size of a program identifier
  uint32_t groupSize =
    nvh::align_up(m_Properties.shaderGroupHandleSize, m_Properties.shaderGroupBaseAlignment);
  uint32_t          groupStride = groupSize;
  vk::DeviceAddress sbtAddress = m_device.getBufferAddress({ m_SBTBuffer.buffer });

  using Stride = vk::StridedDeviceAddressRegionKHR;
  std::array<Stride, 4> strideAddresses{
      Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
      Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 2},  // miss
      Stride{sbtAddress + 3u * groupSize, groupStride, groupSize * 1},  // hit
      Stride{0u, 0u, 0u} };                                             // callable

  cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
    &strideAddresses[3],
    m_outputSize.width, m_outputSize.height,
    1);

  m_debug.endLabel(cmdBuf);
}