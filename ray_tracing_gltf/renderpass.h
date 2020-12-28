#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"

#pragma once
class Renderpass
{
public:
  virtual void setup
  (
    const vk::Device&         device,
    const vk::PhysicalDevice& physicalDevice,
    uint32_t                  queueIndex,
    nvvk::Allocator*          allocator
  );

  virtual void destroy();

  virtual void createPipeline
  (
    vk::DescriptorSetLayout* descSetLayout,
    std::vector<std::string> defaultSearchPaths
  ) = 0;

  vk::RenderPass  m_RenderPass;

protected:
  virtual void createRender(vk::Extent2D size);

  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc{ nullptr };

  vk::Pipeline       m_Pipeline;
  vk::PipelineLayout m_PipelineLayout;

  vk::Extent2D m_outputSize;
};

