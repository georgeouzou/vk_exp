#version 460
#extension GL_NV_ray_tracing : require

struct HitPayload
{
	vec4 color_dist;
};

struct ShadowPayload
{
	float dist;
};

struct TriVertex
{
	vec4 pos;
	vec4 normal;
	vec4 tex_coord;
};

layout(binding = 0) uniform accelerationStructureNV scene;

layout(std430, binding = 3) readonly buffer TriVertices
{
	TriVertex vertices[];
};

layout(std430, binding = 4) readonly buffer TriIndices
{
	uint indices[];
};

layout(binding = 5) uniform sampler2D tex_sampler;

layout(location = 0) rayPayloadInNV HitPayload payload;
					 hitAttributeNV vec2 bary;

layout(location = 1) rayPayloadNV ShadowPayload shadow_payload;

void main()
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);

	uint idp = 3 * gl_PrimitiveID;
	uint vidx0 = indices[idp+0];
	uint vidx1 = indices[idp+1];
	uint vidx2 = indices[idp+2];

	vec2 texc = vertices[vidx0].tex_coord.xy * barys.x +
				vertices[vidx1].tex_coord.xy * barys.y +
				vertices[vidx2].tex_coord.xy * barys.z;

	vec3 norm = vertices[vidx0].normal.xyz * barys.x +
				vertices[vidx1].normal.xyz * barys.y +
				vertices[vidx2].normal.xyz * barys.z;
	
	vec3 hit_color = texture(tex_sampler, texc).bgr;

	const vec3 hit_normal = norm;
	const vec3 hit_pos = gl_WorldRayOriginNV + gl_HitTNV * gl_WorldRayDirectionNV;
	const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	const vec3 to_light1 = normalize(vec3(10.0, 10.0, 10.0));

	const uint shadow_ray_flags = gl_RayFlagsOpaqueNV | gl_RayFlagsTerminateOnFirstHitNV;
	traceNV(scene, shadow_ray_flags, 0xFF, 1, 1, 1, shadow_ray_orig, 0.0, to_light1, 1000.0, 1);
	
	const float ambient = 0.1;
	const float lighting1 = (shadow_payload.dist > 0.0) ? ambient : max(ambient, dot(hit_normal, to_light1));


	vec3 out_color = lighting1 * hit_color;

	payload.color_dist = vec4(out_color, gl_HitTNV);
}