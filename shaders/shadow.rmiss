#version 460
#extension GL_EXT_ray_tracing : require

struct ShadowPayload
{
	float dist;
};

layout(location = 1) rayPayloadInEXT ShadowPayload shadow_payload;

void main()
{
	shadow_payload.dist = -1.0;
}
