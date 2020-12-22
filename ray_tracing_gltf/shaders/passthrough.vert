#version 450
layout (location = 0) out vec2 texCoord;
layout(set = 0, binding = 0) uniform sampler2D noisyTxt;

out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec2 vertex = vec2(-1.0) + vec2(
    float((gl_VertexIndex & 1) << 2),
    float((gl_VertexIndex & 2) << 1));
  gl_Position = vec4(vertex, 0.0, 1.0);
  texCoord = (vertex * 0.5 + vec2(0.5)) * textureSize(noisyTxt, 0);
}
