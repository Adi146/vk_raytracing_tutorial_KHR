#include "postprocessing.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"

void PostProcessing::destroy()
{
  m_device.destroy(m_DescPool);
  m_device.destroy(m_DescSetLayout);

  Renderpass::destroy();
}

void PostProcessing::createRender(vk::Extent2D size, vk::RenderPass renderpass)
{
  Renderpass::createRender(size);

  m_RenderPass = renderpass;
}

void PostProcessing::createDescriptorSet()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_DescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eVertex | vkSS::eFragment));
  m_DescSetLayout = m_DescSetLayoutBind.createLayout(m_device);
  m_DescPool = m_DescSetLayoutBind.createPool(m_device);
  m_DescSet = nvvk::allocateDescriptorSet(m_device, m_DescPool, m_DescSetLayout);
}

void PostProcessing::createPipeline
(
  vk::DescriptorSetLayout* descSetLayout,
  std::vector<std::string> defaultSearchPaths
)
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = { vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstant) };

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_PipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_PipelineLayout,
    m_RenderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths),
    vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/post.frag.spv", true, paths),
    vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_Pipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_Pipeline, "post");
}

void PostProcessing::updateDescriptorSet(VkDescriptorImageInfo* src)
{
  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSet, 0, src));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PostProcessing::draw
(
  const vk::CommandBuffer& cmdBuf
)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, { vk::Viewport(0, 0, (float)m_outputSize.width, (float)m_outputSize.height, 0, 1) });
  cmdBuf.setScissor(0, { {{0, 0}, {m_outputSize.width, m_outputSize.height}} });

  cmdBuf.pushConstants<PushConstant>(m_PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushConstants);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_Pipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_PipelineLayout, 0,
    m_DescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}