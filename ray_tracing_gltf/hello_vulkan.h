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
#include "postprocessing.h"
#include "pathtrace.h"
#include "atrous.h"

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
  PostProcessing m_postprocessing;
  Pathtrace m_pathtrace;
  ATrous m_atrous;

  // #A-Trous
  void updateATrousDescriptorSet();

  // #Post
  void updatePostDescriptorSet();

  // #VKRay
  void updateRtDescriptorSet();
};
