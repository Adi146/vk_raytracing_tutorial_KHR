#include "atrous.h"
#include "nvvk/commands_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvh/fileoperations.hpp"

void ATrous::destroy()
{
  Renderpass::destroy();
  m_device.destroy(m_RenderPass);
  m_device.destroy(m_FramebufferPing);
  m_device.destroy(m_FramebufferPong);

  m_alloc->destroy(m_TexturePong);

  m_device.destroy(m_DescPool);
  m_device.destroy(m_DescSetLayout);
}

void ATrous::createRender(vk::Extent2D size, nvvk::Texture texturePing)
{
  Renderpass::createRender(size);

  m_alloc->destroy(m_TexturePong);
  m_TexturePing = texturePing;

  {
    auto textureCreateInfo = nvvk::makeImage2DCreateInfo(size, m_aTrousFormat,
                                                         vk::ImageUsageFlagBits::eColorAttachment |
                                                         vk::ImageUsageFlagBits::eSampled |
                                                         vk::ImageUsageFlagBits::eStorage);
    nvvk::Image pong = m_alloc->createImage(textureCreateInfo);

    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(pong.image, textureCreateInfo);
    m_TexturePong = m_alloc->createTexture(pong, ivInfo, vk::SamplerCreateInfo());


    m_TexturePong.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  {
    nvvk::CommandPool genCmdBuf(m_device, m_queueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_TexturePong.image, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eGeneral);
    genCmdBuf.submitAndWait(cmdBuf);
  }

  if (!m_RenderPass)
  {
    m_RenderPass =
      nvvk::createRenderPass(m_device, { m_aTrousFormat }, vk::Format::eUndefined, 1, true,
        true, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
  }


  m_device.destroy(m_FramebufferPing);
  m_device.destroy(m_FramebufferPong);

  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_RenderPass);
  info.setAttachmentCount(1);
  info.setWidth(size.width);
  info.setHeight(size.height);
  info.setLayers(1);

  std::vector<vk::ImageView> attachments = { m_TexturePing.descriptor.imageView };
  info.setPAttachments(attachments.data());
  m_FramebufferPing = m_device.createFramebuffer(info);

  attachments = { m_TexturePong.descriptor.imageView };
  info.setPAttachments(attachments.data());
  m_FramebufferPong = m_device.createFramebuffer(info);
}

void ATrous::createDescriptorSet()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_DescSetLayoutBind.addBinding(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment | vkSS::eVertex));
  m_DescSetLayoutBind.addBinding(vkDS(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_DescSetLayoutBind.addBinding(vkDS(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_DescSetLayout = m_DescSetLayoutBind.createLayout(m_device);
  m_DescPool = m_DescSetLayoutBind.createPool(m_device, 2);

  m_DescSetPing = nvvk::allocateDescriptorSet(m_device, m_DescPool, m_DescSetLayout);
  m_DescSetPong = nvvk::allocateDescriptorSet(m_device, m_DescPool, m_DescSetLayout);
}

void ATrous::createPipeline
(
  vk::DescriptorSetLayout* descSetLayout,
  std::vector<std::string> defaultSearchPaths
)
{
  vk::PushConstantRange pushConstantRanges = { vk::ShaderStageFlagBits::eFragment, 0, sizeof(PushConstants) };

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_DescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_PipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_PipelineLayout,
    m_RenderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths, true),
    vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/a-trous.frag.spv", true, paths, true),
    vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_Pipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_Pipeline, "a-trous");
}

void ATrous::updateDesriptorSet
(
  VkDescriptorImageInfo* positionMap,
  VkDescriptorImageInfo* normalMap
)
{
  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPing, 0, &m_TexturePong.descriptor));
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPing, 1, positionMap));
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPing, 2, normalMap));

    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  {
    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPong, 0, &m_TexturePing.descriptor));
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPong, 1, positionMap));
    writes.emplace_back(m_DescSetLayoutBind.makeWrite(m_DescSetPong, 2, normalMap));

    m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }
}

void ATrous::draw(const vk::CommandBuffer& cmdBuf)
{
  vk::ClearValue clearValues[1];
  clearValues[0].setColor(std::array<float, 4>({ 0, 0, 0, 0 }));

  m_debug.beginLabel(cmdBuf, "A-Trous");

  for (int i = 0; i <= 5; i++)
  {
    m_pushConstant.stepwidth = i * 2 + 1;
    m_pushConstant.c_phi = 1.0f / i * m_c_phi0;
    m_pushConstant.n_phi = 1.0f / i * m_n_phi0;
    m_pushConstant.p_phi = 1.0f / i * m_p_phi0;

    vk::RenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.setClearValueCount(1);
    renderPassBeginInfo.setPClearValues(clearValues);
    renderPassBeginInfo.setRenderPass(m_RenderPass);
    renderPassBeginInfo.setFramebuffer((i % 2 == 0) ? m_FramebufferPong : m_FramebufferPing);
    renderPassBeginInfo.setRenderArea({ {}, m_outputSize });

    cmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    cmdBuf.setViewport(0, { vk::Viewport(0, 0, (float)m_outputSize.width, (float)m_outputSize.height, 0, 1) });
    cmdBuf.setScissor(0, { {{0, 0}, {m_outputSize.width, m_outputSize.height}} });

    cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_Pipeline);
    cmdBuf.pushConstants<PushConstants>(m_PipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, m_pushConstant);
    cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_PipelineLayout, 0, (i % 2 == 0) ? m_DescSetPong : m_DescSetPing, {});
    cmdBuf.draw(3, 1, 0, 0);

    cmdBuf.endRenderPass();
  }

  m_debug.endLabel(cmdBuf);
}