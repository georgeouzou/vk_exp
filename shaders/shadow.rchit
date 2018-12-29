#version 460
#extension GL_NV_ray_tracing : require

struct ShadowPayload
{
	float dist;
};

layout(location = 1) rayPayloadInNV ShadowPayload shadow_payload;

void main()
{
	shadow_payload.dist = gl_HitTNV;
}
