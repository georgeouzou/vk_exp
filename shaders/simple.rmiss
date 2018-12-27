#version 460
#extension GL_NV_ray_tracing : require

struct HitInfo
{
	vec4 color_dist;
};

struct Attributes
{
	vec2 bary;
};

layout(location = 0) rayPayloadInNV HitInfo payload;

void main()
{
	uvec2 launch_index = gl_LaunchIDNV.xy;
	vec2 dims = vec2(gl_LaunchSizeNV.xy);
	
	payload.color_dist = vec4(0.0, 0.0, 0.0, 0.0);
}