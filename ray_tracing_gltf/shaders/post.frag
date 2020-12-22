#version 450
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;

layout(push_constant) uniform shaderInformation
{
  int kernelType;
}
pushC;

float kernel3x3[] = float[](
  1.0 / 16, 2.0 / 16, 1.0 / 16,
  2.0 / 16, 4.0 / 16, 2.0 / 16,
  1.0 / 16, 2.0 / 16, 1.0 / 16
);

float kernel5x5[] = float[](
  1.0 / 256, 4.0  / 256, 6.0  / 256, 4.0  / 256, 1.0 / 256,
  4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
  6.0 / 256, 24.0 / 256, 36.0 / 256, 24.0 / 256, 6.0 / 256,
  4.0 / 256, 16.0 / 256, 24.0 / 256, 16.0 / 256, 4.0 / 256,
  1.0 / 256, 4.0  / 256, 6.0  / 256, 4.0  / 256, 1.0 / 256
);

void main()
{
  ivec2 tx = ivec2(texCoord);
  float gamma = 1. / 2.2;

  vec4 color = vec4(0.0);

  if (pushC.kernelType == -1)
  {
    color = texelFetch(noisyTxt, ivec2(tx), 0).rgba;
  }
  else
  {
    int radius = 1;
    if (pushC.kernelType == 0)
    {
      radius = 1;
    }
    else 
    {
      radius = 2;
    }

    int kernelIndex = 0;
    for (int y = -radius; y <= radius; y++) {
      for (int x = -radius; x <= radius; x++) {
        vec4 pixelColor = texelFetch(noisyTxt, ivec2(tx.x + x, tx.y + y), 0).rgba;
        color += pixelColor * (pushC.kernelType == 0 ? kernel3x3[kernelIndex] : kernel5x5[kernelIndex]);
        kernelIndex++;
      }
    }
  }

  fragColor = pow(color, vec4(gamma));
}
