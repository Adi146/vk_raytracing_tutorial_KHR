/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "hello_vulkan.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Extra UI
void renderUI(HelloVulkan& helloVk, nvmath::vec4f* clearColor)
{
  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Common");
  ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(clearColor));
  ImGui::SliderInt("Samples", &helloVk.m_pathtrace.m_pushConstants.samples, 1, 16);
  ImGui::SliderInt("Bounces", &helloVk.m_pathtrace.m_pushConstants.bounces, 0, 5);
  ImGui::SliderInt("Samples Per Bounce", &helloVk.m_pathtrace.m_pushConstants.bounceSamples, 1, 4);

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Direct Lighting");
  ImGui::RadioButton("Off", &helloVk.m_pathtrace.m_pushConstants.lightType, -1);
  ImGui::SameLine();
  ImGui::RadioButton("Point", &helloVk.m_pathtrace.m_pushConstants.lightType, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Infinite", &helloVk.m_pathtrace.m_pushConstants.lightType, 1);

  if(helloVk.m_pathtrace.m_pushConstants.lightType != -1)
  {
    ImGui::SliderFloat3("Light Position", &helloVk.m_pathtrace.m_pushConstants.lightPosition.x, -20.f, 20.f);
    ImGui::SliderFloat("Light Intensity", &helloVk.m_pathtrace.m_pushConstants.lightIntensity, 0.f, 100.f);
  }

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Temporal Filter");
  ImGui::SliderFloat("Alpha", &helloVk.m_pathtrace.m_pushConstants.temporalAlpha, 0.f, 0.99f);

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Post Processing");
  ImGui::RadioButton("Blur Off", &helloVk.m_postprocessing.m_pushConstants.kernelType, -1);
  ImGui::SameLine();
  ImGui::RadioButton("Gaussian Blur 3x3", &helloVk.m_postprocessing.m_pushConstants.kernelType, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Gaussian Blur 5x5", &helloVk.m_postprocessing.m_pushConstants.kernelType, 1);

  ImGui::TextColored(ImVec4(1, 1, 0, 1), "Denoising");
  ImGui::RadioButton("Denoiser Off", &helloVk.m_denoiserKind, -1);
  ImGui::SameLine();
  ImGui::RadioButton("A-Trous", &helloVk.m_denoiserKind, 0);
  ImGui::SameLine();
  ImGui::RadioButton("OptiX", &helloVk.m_denoiserKind, 1);

  if (helloVk.m_denoiserKind == 0)
  {
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "A-Trous");
    ImGui::SliderFloat("C_Phi", &helloVk.m_atrous.m_c_phi0, 0.0f, 1.0f);
    ImGui::SliderFloat("N_Phi", &helloVk.m_atrous.m_n_phi0, 0.0f, 100.0f);
    ImGui::SliderFloat("P_Phi", &helloVk.m_atrous.m_p_phi0, 0.0f, 100.0f);
  }

  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
              ImGui::GetIO().Framerate);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1920;
static int const SAMPLE_HEIGHT = 1080;

//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  UNUSED(argc);

  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window =
      glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat(nvmath::vec3f(0, 0, 15), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(argv[0], PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      PROJECT_ABSDIRECTORY,
      PROJECT_ABSDIRECTORY "..",
      NVPSystem::exePath(),
      NVPSystem::exePath() + std::string(PROJECT_NAME),
  };


  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo(true);
  contextInfo.setVersion(1, 2);
  contextInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  contextInfo.addInstanceLayer("VK_LAYER_KHRONOS_validation");
  contextInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);
#ifdef WIN32
  contextInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  contextInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  contextInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  contextInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
  vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature;
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false,
                                 &accelFeature);
  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature;
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false,
                                 &rtPipelineFeature);
  // #OptiX
  contextInfo.addInstanceExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  contextInfo.addInstanceExtension(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineFeature;
  contextInfo.addDeviceExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, false, &timelineFeature);

  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);


  // Create example
  HelloVulkan helloVk;

  helloVk.m_denoiser.initOptiX();

  // Window need to be opened to get the surface on which to draw
  const vk::SurfaceKHR surface = helloVk.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  helloVk.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice,
                vkctx.m_queueGCT.familyIndex);
  helloVk.createSwapchain(surface, SAMPLE_WIDTH, SAMPLE_HEIGHT);
  helloVk.createDepthBuffer();
  helloVk.createRenderPass();
  helloVk.createFrameBuffers();

  // Setup Imgui
  helloVk.initGUI(0);  // Using sub-pass 0

  // Creation of the example
  helloVk.loadScene(nvh::findFile("media/scenes/cornellBox.gltf", defaultSearchPaths));
  //helloVk.loadScene(nvh::findFile("../glTF-Sample-Models/2.0/Sponza/glTF/Sponza.gltf", defaultSearchPaths));

  helloVk.createDenoiseOutImage();
  helloVk.m_denoiser.allocateBuffers(helloVk.getSize());

  helloVk.m_pathtrace.createRender(helloVk.getSize());

  helloVk.createDescriptorSetLayout();
  helloVk.createUniformBuffer();
  helloVk.updateDescriptorSet();

  helloVk.m_gbuffer.createRender(helloVk.getSize());
  helloVk.m_gbuffer.createPipeline(&helloVk.m_descSetLayout, defaultSearchPaths);

  helloVk.m_atrous.createRender(helloVk.getSize(), helloVk.m_pathtrace.m_outputColor);
  helloVk.m_atrous.createDescriptorSet();
  helloVk.m_atrous.createPipeline(&helloVk.m_atrous.m_DescSetLayout, defaultSearchPaths);
  helloVk.m_atrous.updateDesriptorSet(&helloVk.m_gbuffer.m_position.descriptor, &helloVk.m_gbuffer.m_normal.descriptor);

  // #VKRay
  helloVk.m_pathtrace.createBottomLevelAS(helloVk.m_gltfScene, helloVk.m_vertexBuffer.buffer, helloVk.m_indexBuffer.buffer);
  helloVk.m_pathtrace.createTopLevelAS(helloVk.m_gltfScene);
  helloVk.m_pathtrace.createDescriptorSet(helloVk.m_rtPrimLookup.buffer);
  helloVk.m_pathtrace.createPipeline(&helloVk.m_descSetLayout, defaultSearchPaths);
  helloVk.m_pathtrace.createShaderBindingTable();

  helloVk.m_postprocessing.createRender(helloVk.getSize(), helloVk.getRenderPass());
  helloVk.m_postprocessing.createDescriptorSet();
  helloVk.m_postprocessing.createPipeline(&helloVk.m_postprocessing.m_DescSetLayout, defaultSearchPaths);


  nvmath::vec4f clearColor = nvmath::vec4f(1, 1, 1, 1.00f);

  helloVk.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(helloVk.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Show UI window.
    renderUI(helloVk, &clearColor);

    // Start rendering the scene
    helloVk.prepareFrame();

    switch (helloVk.m_denoiserKind)
    {
      case 0:
        helloVk.m_postprocessing.updateDescriptorSet(&helloVk.m_atrous.m_TexturePong.descriptor);
        break;
      case 1:
        helloVk.m_postprocessing.updateDescriptorSet(&helloVk.m_imageOut.descriptor);
        break;
      default:
        helloVk.m_postprocessing.updateDescriptorSet(&helloVk.m_pathtrace.m_outputColor.descriptor);
    }

    // Start command buffer of this frame
    auto                     curFrame = helloVk.getCurFrame();
    const vk::CommandBuffer& cmdBuf   = helloVk.getCommandBuffers()[curFrame];

    cmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Clearing screen
    vk::ClearValue clearValues[2];
    clearValues[0].setColor(
        std::array<float, 4>({clearColor[0], clearColor[1], clearColor[2], clearColor[3]}));
    clearValues[1].setDepthStencil({1.0f, 0});

    if (helloVk.m_denoiserKind == 1) // OptiX
    {
      nvvk::CommandPool cmdPool(helloVk.getDevice(), helloVk.getQueueFamily());
      auto subCmdBuf = cmdPool.createCommandBuffer();

      // Updating camera buffer
      helloVk.updateUniformBuffer(subCmdBuf);

      helloVk.m_gbuffer.draw
      (
        subCmdBuf,
        helloVk.m_descSet,
        std::vector<vk::Buffer>
        {
          helloVk.m_vertexBuffer.buffer,
          helloVk.m_normalBuffer.buffer,
          helloVk.m_uvBuffer.buffer
        },
        helloVk.m_indexBuffer.buffer,
        helloVk.m_gltfScene
      );

      helloVk.m_pathtrace.draw(subCmdBuf, helloVk.m_descSet, clearColor);

      helloVk.m_denoiser.imageToBuffer
      (
        subCmdBuf,
        {
          helloVk.m_pathtrace.m_outputColor,
          helloVk.m_gbuffer.m_color,
          helloVk.m_gbuffer.m_normal
        }
      );
      helloVk.m_denoiser.submitWithSemaphore(subCmdBuf, helloVk.m_fenceValue);
      helloVk.m_denoiser.denoiseImageBuffer(subCmdBuf, &helloVk.m_imageOut, helloVk.m_fenceValue);
      helloVk.m_denoiser.waitSemaphore(helloVk.m_fenceValue);
      helloVk.m_denoiser.bufferToImage(cmdBuf, &helloVk.m_imageOut);
    }
    else
    {
      // Updating camera buffer
      helloVk.updateUniformBuffer(cmdBuf);

      helloVk.m_gbuffer.draw
      (
        cmdBuf,
        helloVk.m_descSet,
        std::vector<vk::Buffer>
      {
        helloVk.m_vertexBuffer.buffer,
          helloVk.m_normalBuffer.buffer,
          helloVk.m_uvBuffer.buffer
      },
        helloVk.m_indexBuffer.buffer,
          helloVk.m_gltfScene
          );

      helloVk.m_pathtrace.draw(cmdBuf, helloVk.m_descSet, clearColor);
    }

    if(helloVk.m_denoiserKind == 0) // A-Trous
    {
      helloVk.m_atrous.draw(cmdBuf);
    }

    // 2nd rendering pass: tone mapper, UI
    {
      vk::RenderPassBeginInfo postRenderPassBeginInfo;
      postRenderPassBeginInfo.setClearValueCount(2);
      postRenderPassBeginInfo.setPClearValues(clearValues);
      postRenderPassBeginInfo.setRenderPass(helloVk.m_postprocessing.m_RenderPass);
      postRenderPassBeginInfo.setFramebuffer(helloVk.getFramebuffers()[curFrame]);
      postRenderPassBeginInfo.setRenderArea({{}, helloVk.getSize()});

      cmdBuf.beginRenderPass(postRenderPassBeginInfo, vk::SubpassContents::eInline);
      // Rendering tonemapper
      helloVk.m_postprocessing.draw(cmdBuf);
      // Rendering UI
      ImGui::Render();
      // Rendering UI
      ImGui::RenderDrawDataVK(cmdBuf, ImGui::GetDrawData());
      cmdBuf.endRenderPass();
    }

    // Submit for display
    cmdBuf.end();
    helloVk.submitFrame();
  }

  // Cleanup
  helloVk.getDevice().waitIdle();
  helloVk.destroyResources();
  helloVk.destroy();

  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
