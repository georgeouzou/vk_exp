#version 460
#extension GL_NV_ray_tracing : require

struct HitInfo
{
	vec4 color_dist;
};

#if 0
struct STriVertex
{
	vec3 pos;
	vec3 color;
	vec2 tex_coord;
};

layout(std440, binding = 2) buffer vbo
{
	STriVertex vertices[];
};
#endif

layout(location = 0) rayPayloadNV HitInfo payload;

hitAttributeNV vec2 bary;

void main()
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);

	const vec3 A = vec3(1.0, 0.0, 0.0);
	const vec3 B = vec3(0.0, 1.0, 0.0);
	const vec3 C = vec3(0.0, 0.0, 1.0);
	vec3 hit_color = A * barys.x + B * barys.y + C * barys.z;
	payload.color_dist = vec4(hit_color, 0.0);
}