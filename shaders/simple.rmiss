#version 460
#extension GL_NV_ray_tracing : require

struct HitPayload
{
	vec4 color_dist;
	int depth;
};

struct Attributes
{
	vec2 bary;
};

layout(location = 0) rayPayloadInNV HitPayload payload;

void main()
{
	//uvec2 launch_index = gl_LaunchIDNV.xy;
	//vec2 dims = vec2(gl_LaunchSizeNV.xy);
	
	payload.color_dist = vec4(1.0, 0.7, 0.5, -1.0);
}