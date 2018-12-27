#version 460
#extension GL_NV_ray_tracing : require

struct HitInfo
{
	vec4 color_dist;
};

struct TriVertex
{
	vec4 pos;
	vec4 color;
	vec4 tex_coord;
};

layout(std430, binding = 3) readonly buffer TriVertices
{
	TriVertex vertices[];
};

layout(std430, binding = 4) readonly buffer TriIndices
{
	uint indices[];
};

layout(binding = 5) uniform sampler2D tex_sampler;

layout(location = 0) rayPayloadInNV HitInfo payload;
					 hitAttributeNV vec2 bary;

void main()
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);

	uint idp = 3 * gl_PrimitiveID;
	uint vidx0 = indices[idp+0];
	uint vidx1 = indices[idp+1];
	uint vidx2 = indices[idp+2];
	
#if 0
	vec3 hit_color = vertices[vidx0].color.xyz * barys.x +
					 vertices[vidx1].color.xyz * barys.y +
					 vertices[vidx2].color.xyz * barys.z;
#endif
	vec2 texc = vertices[vidx0].tex_coord.xy * barys.x +
				vertices[vidx1].tex_coord.xy * barys.y +
				vertices[vidx2].tex_coord.xy * barys.z;

	vec3 hit_color = texture(tex_sampler, texc).rgb;

	payload.color_dist = vec4(hit_color, 0.0);
}