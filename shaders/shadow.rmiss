#version 460
#extension GL_EXT_ray_tracing : require

#include "common.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload shadow_payload;

void main()
{
	shadow_payload.in_shadow = -1.0;
}
