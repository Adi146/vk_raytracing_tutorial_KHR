#pragma once
#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvh/gltfscene.hpp"

class GBuffer
{
public:
  void setup
  (
    const vk::Device&         device,
    const vk::PhysicalDevice& physicalDevice,
    uint32_t                  queueIndex,
    nvvk::Allocator*          allocator
  );

  void destroy();
  void createRender(vk::Extent2D size);
  void createPipeline(vk::DescriptorSetLayout* descSetLayout,
                      std::vector<std::string> defaultSearchPaths);
  void draw(
    const vk::CommandBuffer& cmdBuf,
    vk::DescriptorSet        descSet,
    std::vector<vk::Buffer>  vertexBuffers,
    vk::Buffer               indexBuffer,
    nvh::GltfScene&          gltfScene
  );

  nvvk::Texture m_position;
  nvvk::Texture m_normal;
  nvvk::Texture m_color;
  nvvk::Texture m_depth;

  vk::RenderPass  m_RenderPass;
  vk::Framebuffer m_Framebuffer;

private:
  struct ObjPushConstant
  {
    int instanceId{0};  // To retrieve the transformation matrix
    int materialId{0};
  };
  ObjPushConstant m_pushConstant;

  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc{nullptr};

  vk::Pipeline       m_Pipeline;
  vk::PipelineLayout m_PipelineLayout;

  vk::Format         m_positionColorFormat{vk::Format::eR32G32B32A32Sfloat};
  vk::Format         m_normalColorFormat{vk::Format::eR32G32B32A32Sfloat};
  vk::Format         m_colorColorFormat{vk::Format::eR8G8B8A8Unorm};
  vk::Format         m_depthColorFormat{vk::Format::eD32Sfloat};

  vk::Extent2D m_outputSize;
};
