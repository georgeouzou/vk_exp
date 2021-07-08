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

vec3 random_in_unit_sphere(inout uint seed)
{
	vec3 p;
    uint loops = 16;
    while (loops-- > 0u) {
	    float x = random_float(seed);
	    float y = random_float(seed);
	    float z = random_float(seed);
	    p = 2.0 * vec3(x, y, z) - vec3(1.0);
        if (length(p) < 1) break;
    }
    return p;
}

vec3 random_unit_vector(inout uint seed)
{
    return normalize(random_in_unit_sphere(seed));
}

vec3 random_in_hemisphere(inout uint seed, vec3 normal)
{
    vec3 in_sphere = random_in_unit_sphere(seed);
    bool same_hemisphere = dot(in_sphere, normal) > 0.0;
    in_sphere *= same_hemisphere ? 1.0 : -1.0;
    return in_sphere;
}


#endif //RANDOM_H_GLSL