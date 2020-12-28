#pragma once
#include "renderpass.h"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvh/gltfscene.hpp"

class Pathtrace : public Renderpass
{
public:
  void setup
  (
    const vk::Device&         device,
    const vk::PhysicalDevice& physicalDevice,
    uint32_t                  queueIndex,
    nvvk::Allocator*          allocator
  )override;
  void destroy() override;

  void createRender(vk::Extent2D size);
  void createDescriptorSet(vk::Buffer primitiveLookupBuffer);
  void createPipeline
  (
    vk::DescriptorSetLayout* descSetLayout,
    std::vector<std::string> defaultSearchPaths
  ) override;
  void createShaderBindingTable();

  void createBottomLevelAS
  (
    const nvh::GltfScene& gltfScene,
    const vk::Buffer vertexBuffer,
    const vk::Buffer indexBuffer
  );
  void createTopLevelAS(const nvh::GltfScene& gltfScene);

  void updateDescriptorSet();

  void draw
  (
    const vk::CommandBuffer& cmdBuf,
    vk::DescriptorSet        descSet,
    const nvmath::vec4f&     clearColor
  );

  nvvk::Texture m_outputColor;
  nvvk::Texture m_historyColor;
  nvvk::Texture m_depth;

  vk::Framebuffer m_Framebuffer;

  nvvk::DescriptorSetBindings m_DescSetLayoutBind;
  vk::DescriptorSetLayout     m_DescSetLayout;
  vk::DescriptorSet           m_DescSet;

  struct PushConstant
  {
    nvmath::vec4f clearColor;
    nvmath::vec3f lightPosition{ 0.f, 4.5f, 0.f };
    float         lightIntensity{ 10.f };
    int           lightType{ -1 }; // -1: off, 0: point, 1: infinite
    int           frame{ 0 };
    int           samples{ 2 };
    int           bounces{ 2 };
    int           bounceSamples{ 2 };
    float         temporalAlpha{ 0.1f };
  } m_pushConstants;

private:
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry
  (
    const nvh::GltfPrimMesh& prim,
    const vk::Buffer vertexBuffer,
    const vk::Buffer indexBuffer
  );

  vk::Format                  m_colorFormat{ vk::Format::eR32G32B32A32Sfloat };
  vk::Format                  m_depthFormat{ vk::Format::eD32Sfloat };

  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR   m_Properties;
  nvvk::RaytracingBuilderKHR                          m_Builder;
  vk::DescriptorPool                                  m_DescPool;

  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_ShaderGroups;
  nvvk::Buffer                                        m_SBTBuffer;
};

