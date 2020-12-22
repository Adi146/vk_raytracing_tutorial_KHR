#version 450
layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;
layout(set = 0, binding = 1) uniform sampler2D positionTxt;
layout(set = 0, binding = 2) uniform sampler2D normalTxt;
layout(set = 0, binding = 3) uniform sampler2D colorTxt;

layout(push_constant) uniform shaderInformation
{
    int stepwidth;
    float c_phi;
    float n_phi;
    float p_phi;
}
pushC;

#define KERNEL_SIZE 25

float kernel[KERNEL_SIZE] = float[](
    1.0 / 256, 1.0 / 64, 3.0 / 128, 1.0 / 64, 1.0 / 256,
    1.0 / 64,  1.0 / 16, 3.0 / 32,  1.0 / 16, 1.0 / 64,
    3.0 / 128, 3.0 / 32, 9.0 / 64,  3.0 / 32, 3.0 / 128,
    1.0 / 64,  1.0 / 16, 3.0 / 32,  1.0 / 16, 1.0 / 64,
    1.0 / 256, 1.0 / 64, 3.0 / 128, 1.0 / 64, 1.0 / 256
);

void main(void) {
    vec4 sum = vec4(0.0);

    ivec2 tx = ivec2(texCoord);
    vec4 cval = texelFetch(noisyTxt, tx, 0).rgba;
    vec3 nval = texelFetch(normalTxt, tx, 0).rgb;
    vec3 pval = texelFetch(positionTxt, tx, 0).rgb;

    float sw2 = pushC.stepwidth * pushC.stepwidth;

    float cum_w = 0.0;

    int radius = 2;
    int kernelIndex = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            ivec2 uv = tx + ivec2(x,y) * pushC.stepwidth;

            vec4 ctmp = texelFetch(noisyTxt, uv, 0);
            vec3 ntmp = texelFetch(normalTxt, uv, 0).xyz;
            vec3 ptmp = texelFetch(positionTxt, uv, 0).xyz;      

            //float n_w = dot(nval, ntmp);
            //if (n_w < 1E-3)
            //    continue;
            vec3 nt = nval - ntmp;
            float n_w = min(exp(-(max(dot(nt,nt)/(sw2),0.0)) / pushC.n_phi), 1.0);

            vec4 ct = cval - ctmp;
            //float c_w = clamp(1.0 - dot(ct, ct) / pushC.c_phi * sw2, 0.0, 1.0);
            float c_w = min(exp(-(dot(ct, ct)) / pushC.c_phi), 1.0);

            vec3 pt = pval - ptmp;
            //float p_w = clamp(1.0 - dot(pt, pt) / pushC.p_phi, 0.0, 1.0);
            float p_w = min(exp(-(dot(pt, pt)) / pushC.p_phi), 1.0);
       
            float weight = c_w * kernel[kernelIndex];
            sum += ctmp * weight;
            cum_w += weight;

            kernelIndex++;
        }
    }

    fragColor = sum / cum_w;
}

/*float computeVariance()
{
    float sum = 0.0;

    const float kernel [2][2] = {
        { 1.0 / 4.0, 1.0/ 8.0 },
        { 1.0 / 8.0, 1.0/ 4.0 }
    }

    int radius = 1
    for (int  y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            ivec2 uv = ivec2(texCoord) + ivec2(x, y);

            float k = kernel[abs(x)][abs(y)];

            sum += 
        }
    }
}*/