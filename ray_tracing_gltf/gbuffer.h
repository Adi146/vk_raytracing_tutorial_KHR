#pragma once
#include "renderpass.h"
#include "nvh/gltfscene.hpp"

class GBuffer : public Renderpass
{
public:
  void destroy() override;

  void createRender(vk::Extent2D size);
  void createPipeline
  (
    vk::DescriptorSetLayout* descSetLayout,
    std::vector<std::string> defaultSearchPaths
  ) override;

  void draw
  (
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

  vk::Framebuffer m_Framebuffer;

private:
  struct PushConstant
  {
    int instanceId{0};  // To retrieve the transformation matrix
    int materialId{0};
  } m_pushConstant;

  vk::Format         m_positionColorFormat{vk::Format::eR32G32B32A32Sfloat};
  vk::Format         m_normalColorFormat{vk::Format::eR32G32B32A32Sfloat};
  vk::Format         m_colorColorFormat{vk::Format::eR8G8B8A8Unorm};
  vk::Format         m_depthColorFormat{vk::Format::eD32Sfloat};
};
