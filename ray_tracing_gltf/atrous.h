#pragma once
#include "renderpass.h"
#include "nvvk/descriptorsets_vk.hpp"

class ATrous : public Renderpass
{
public:
  void destroy() override;

  void createRender(vk::Extent2D size, nvvk::Texture texturePing);
  void createDescriptorSet();
  void createPipeline
  (
    vk::DescriptorSetLayout* descSetLayout,
    std::vector<std::string> defaultSearchPaths
  ) override;

  void updateDesriptorSet
  (
    VkDescriptorImageInfo* positionMap,
    VkDescriptorImageInfo* normalMap
  );

  void draw
  (
    const vk::CommandBuffer& cmdBuf
  );

  nvvk::Texture m_TexturePing;
  nvvk::Texture m_TexturePong;

  vk::Framebuffer m_FramebufferPing;
  vk::Framebuffer m_FramebufferPong;

  nvvk::DescriptorSetBindings m_DescSetLayoutBind;
  vk::DescriptorSetLayout     m_DescSetLayout;
  vk::DescriptorSet           m_DescSetPing;
  vk::DescriptorSet           m_DescSetPong;

  bool m_enabled = false;
  float m_c_phi0 = 1E-2f;
  float m_n_phi0 = 1E-2f;
  float m_p_phi0 = 1E-1f;

private:
  struct PushConstants
  {
    int stepwidth;
    float c_phi;
    float n_phi;
    float p_phi;
  } m_pushConstant;

  vk::DescriptorPool          m_DescPool;

  vk::Format                  m_aTrousFormat{ vk::Format::eR32G32B32A32Sfloat };
};

