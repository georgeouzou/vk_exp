#version 460
#extension GL_EXT_ray_tracing : require

struct ShadowPayload
{
	float in_shadow;
};

layout(location = 1) rayPayloadInEXT ShadowPayload shadow_payload;

void main()
{
	shadow_payload.in_shadow = gl_HitTEXT;
}
