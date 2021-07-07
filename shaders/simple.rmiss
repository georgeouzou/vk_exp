#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main()
{	
	payload.color_dist = vec4(0.5, 0.8, 0.9, -1.0);
}
