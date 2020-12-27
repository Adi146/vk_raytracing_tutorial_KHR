#pragma once
#include "renderpass.h"
#include "nvvk/descriptorsets_vk.hpp"

class PostProcessing : public Renderpass
{
public:
  void destroy() override;

  void createRender(vk::Extent2D size, vk::RenderPass renderpass);
  void createDescriptorSet();
  void createPipeline
  (
    vk::DescriptorSetLayout* descSetLayout,
    std::vector<std::string> defaultSearchPaths
  ) override;

  void draw
  (
    const vk::CommandBuffer& cmdBuf
  );

  nvvk::DescriptorSetBindings m_DescSetLayoutBind;
  vk::DescriptorSet           m_DescSet;
  vk::DescriptorSetLayout     m_DescSetLayout;

  struct PushConstant
  {
    int kernelType{ -1 }; // -1: off, 0: Gaussian Blur 3x3, 1: Gaussian Blur 5x5
  } m_pushConstants;

private:
  vk::DescriptorPool          m_DescPool;
};

