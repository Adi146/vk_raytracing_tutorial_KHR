#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#include "binding.glsl"
#include "gltf.glsl"

layout(push_constant) uniform shaderInformation
{
  uint  instanceId;
  int   materialId;
} pushC;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;

layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 viewDir;
layout(location = 4) in vec3 worldPos;

layout(set = 0, binding = B_MATERIALS) buffer _GltfMaterial { GltfShadeMaterial materials[]; };
layout(set = 0, binding = B_TEXTURES) uniform sampler2D[] textureSamplers;

void main() 
{
    GltfShadeMaterial mat = materials[nonuniformEXT(pushC.materialId)];

    vec3  albedo    = mat.pbrBaseColorFactor.xyz;
    if(mat.pbrBaseColorTexture > -1)
    {
      uint txtId = mat.pbrBaseColorTexture;
      albedo *= texture(textureSamplers[nonuniformEXT(txtId)], fragTexCoord).xyz;
    }

	outPosition = vec4(worldPos, 0.0);
	outNormal = vec4(fragNormal, 0.0);
	outAlbedo = vec4(albedo, 0.0);
}