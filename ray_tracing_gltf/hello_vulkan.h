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
#pragma once
#include <vulkan/vulkan.hpp>

#define NVVK_ALLOC_DEDICATED
#include "nvvk/allocator_vk.hpp"
#include "nvvk/appbase_vkpp.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

// #VKRay
#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"

#include "gbuffer.h"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBase
{
public:
  void setup(const vk::Instance&       instance,
             const vk::Device&         device,
             const vk::PhysicalDevice& physicalDevice,
             uint32_t                  queueFamily) override;
  void createDescriptorSetLayout();
  void loadScene(const std::string& filename);
  void updateDescriptorSet();
  void createUniformBuffer();
  void createTextureImages(const vk::CommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuf);
  void onResize(int /*w*/, int /*h*/) override;
  void destroyResources();

  // Structure used for retrieving the primitive information in the closest hit
  // The gl_InstanceCustomIndexNV
  struct RtPrimitiveLookup
  {
    uint32_t indexOffset;
    uint32_t vertexOffset;
    int      materialIndex;
  };


  nvh::GltfScene m_gltfScene;
  nvvk::Buffer   m_vertexBuffer;
  nvvk::Buffer   m_normalBuffer;
  nvvk::Buffer   m_uvBuffer;
  nvvk::Buffer   m_indexBuffer;
  nvvk::Buffer   m_materialBuffer;
  nvvk::Buffer   m_matrixBuffer;
  nvvk::Buffer   m_rtPrimLookup;

  // Information pushed at each draw call
  struct ObjPushConstant
  {
    int           instanceId{0};  // To retrieve the transformation matrix
    int           materialId{0};
  };
  ObjPushConstant m_pushConstant;

  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  vk::DescriptorPool          m_descPool;
  vk::DescriptorSetLayout     m_descSetLayout;
  vk::DescriptorSet           m_descSet;

  nvvk::Buffer               m_cameraMat;  // Device-Host of the camera matrices
  std::vector<nvvk::Texture> m_textures;   // vector of all textures of the scene

  nvvk::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil          m_debug;  // Utility to name objects

  GBuffer m_gbuffer;

  // #A-Trous
  void createATrousRender();
  void createATrousPipeline();
  void createATrousDescriptor();
  void updateATrousDescriptorSet();
  void drawATrous(const vk::CommandBuffer& cmdBuf);

  nvvk::DescriptorSetBindings m_aTrousDescSetLayoutBind;
  vk::DescriptorPool          m_aTrousDescPool;
  vk::DescriptorSetLayout     m_aTrousDescSetLayout;
  vk::DescriptorSet           m_aTrousDescSetPing;
  vk::DescriptorSet           m_aTrousDescSetPong;
  vk::Pipeline                m_aTrousPipeline;
  vk::PipelineLayout          m_aTrousPipelineLayout;
  vk::RenderPass              m_aTrousRenderPass;
  vk::Framebuffer             m_aTrousFramebufferPing;
  vk::Framebuffer             m_aTrousFramebufferPong;
  nvvk::Texture               m_aTrousTexturePing;
  nvvk::Texture               m_aTrousTexturePong;

  struct ATrousPushConstants
  {
    int stepwidth;
    float c_phi;
    float n_phi;
    float p_phi;
  };
  ATrousPushConstants m_aTrousPushConstant;

  bool m_enableATrous = false;
  float m_c_phi0 = 1E-2f;
  float m_n_phi0 = 1E-2f;
  float m_p_phi0 = 1E-1f;

  // #Post
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(vk::CommandBuffer cmdBuf);

  struct PostPushConstant
  {
    int kernelType{-1}; // -1: off, 0: Gaussian Blur 3x3, 1: Gaussian Blur 5x5
  } m_postPushConstants;

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  vk::DescriptorPool          m_postDescPool;
  vk::DescriptorSetLayout     m_postDescSetLayout;
  vk::DescriptorSet           m_postDescSet;
  vk::Pipeline                m_postPipeline;
  vk::PipelineLayout          m_postPipelineLayout;
  vk::RenderPass              m_offscreenRenderPass;
  vk::Framebuffer             m_offscreenFramebuffer;
  nvvk::Texture               m_offscreenColor;
  vk::Format                  m_offscreenColorFormat{vk::Format::eR32G32B32A32Sfloat};
  nvvk::Texture               m_offscreenDepth;
  vk::Format                  m_offscreenDepthFormat{vk::Format::eD32Sfloat};

  // #VKRay
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::GltfPrimMesh& prim);

  void initRayTracing();
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void createRtShaderBindingTable();
  void raytrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);
  void updateFrame();
  void resetFrame();

  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR   m_rtProperties;
  nvvk::RaytracingBuilderKHR                          m_rtBuilder;
  nvvk::DescriptorSetBindings                         m_rtDescSetLayoutBind;
  vk::DescriptorPool                                  m_rtDescPool;
  vk::DescriptorSetLayout                             m_rtDescSetLayout;
  vk::DescriptorSet                                   m_rtDescSet;
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  vk::PipelineLayout                                  m_rtPipelineLayout;
  vk::Pipeline                                        m_rtPipeline;
  nvvk::Buffer                                        m_rtSBTBuffer;

  struct RtPushConstant
  {
    nvmath::vec4f clearColor;
    nvmath::vec3f lightPosition{0.f, 4.5f, 0.f};
    float         lightIntensity{10.f};
    int           lightType{-1}; // -1: off, 0: point, 1: infinite
    int           frame{0};
    int           samples{2};
    int           bounces{2};
    int           bounceSamples{2};
    float         temporalAlpha{0.1f};
  } m_rtPushConstants;
};
