#version 460
#extension GL_NV_ray_tracing : require

struct HitInfo
{
	vec4 color_dist;
};

struct TriVertex
{
	vec3 pos;
	float pad;
	vec3 color;
	float pad1;
	vec2 tex_coord;
	vec2 pad2;
};

layout(std430, binding = 3) readonly buffer TriVertices
{
	TriVertex vertices[];
};


layout(location = 0) rayPayloadInNV HitInfo payload;
					 hitAttributeNV vec2 bary;

void main()
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);

	uint id = 3 * gl_PrimitiveID;
	vec3 hit_color = vertices[id+0].color.xyz * barys.x +
					 vertices[id+1].color.xyz * barys.y +
					 vertices[id+2].color.xyz * barys.z;
		
	payload.color_dist = vec4(hit_color, 0.0);
}