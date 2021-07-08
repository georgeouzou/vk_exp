#ifndef RANDOM_H_GLSL
#define RANDOM_H_GLSL

// random functions taken from: 
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/tree/master/ray_tracing_jitter_cam

// Generate a random unsigned int from two unsigned int values
// See Zafar, Olano, and Curtis, "GPU Random Numbers via the Tiny Encryption Algorithm"
uint random_tea(uint val0, uint val1)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0u;

    for(uint n = 0u; n < 16u; n++) {
        s0 += 0x9E3779B9u;
        v0 += ((v1 << 4u) + 0xA341316Cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xC8013EA4u);
        v1 += ((v0 << 4u) + 0xAD90777Du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7E95761Eu);
    }

    return v0;
}

// Generate a random unsigned int in [0, 2^24)
// using the Numerical Recipes linear congruential generator
uint random_lcg(inout uint seed)
{
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    seed = (LCG_A * seed + LCG_C);
    return seed & 0x00FFFFFFu;
}

// Generate a random float in [0, 1)
float random_float(inout uint seed) 
{
    return (float(random_lcg(seed)) / float(0x01000000));
}

#endif //RANDOM_H_GLSL