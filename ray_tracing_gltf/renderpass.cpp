#include "renderpass.h"

void Renderpass::setup
(
  const vk::Device&         device,
  const vk::PhysicalDevice& physicalDevice,
  uint32_t                  queueIndex,
  nvvk::Allocator*          allocator
)
{
  m_device     = device;
  m_debug.setup(device);
  m_queueIndex = queueIndex;
  m_alloc      = allocator;
}

void Renderpass::destroy()
{
  m_device.destroy(m_Pipeline);
  m_device.destroy(m_PipelineLayout);
}

void Renderpass::createRender(vk::Extent2D size)
{
  m_outputSize = size;
}