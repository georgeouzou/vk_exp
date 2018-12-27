#version 460
#extension GL_NV_ray_tracing : require

struct ShadowInfo
{
	float distance;
};

layout(location = 1) rayPayloadInNV ShadowInfo shadow_payload;

void main()
{
	shadow_payload.distance = gl_HitTNV;
}
