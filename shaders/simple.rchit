#version 460
#extension GL_EXT_ray_tracing : require

struct HitPayload
{
	vec4 color_dist;
	int depth;
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;

layout(binding = 2) uniform GlobalUniforms
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 iview;
	mat4 iproj;
	vec4 light_pos;
} ubo;

layout(std430, binding = 3) readonly buffer TriVertices
{
	TriVertex vertices[];
};

layout(std430, binding = 4) readonly buffer TriIndices
{
	uint indices[];
};

layout(location = 0) rayPayloadInEXT HitPayload payload;
					 hitAttributeEXT vec2 bary;

layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;

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
	norm = normalize(norm);
	
	vec3 hit_color = vec3(0.7, 0.7, 0.7);

	const vec3 hit_normal = norm;
	const vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	const vec3 to_light1 = normalize(ubo.light_pos.xyz);

	const uint shadow_ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
	traceRayEXT(scene, shadow_ray_flags, 0xFF, 1, 1, 1, shadow_ray_orig, 0.001, to_light1, 1000.0, 1);

	const float ambient = 0.1;
	const float lighting1 = (shadow_payload.dist > 0.0) ? ambient : max(ambient, dot(hit_normal, to_light1));
	vec3 out_color = lighting1 * hit_color;
	payload.color_dist = vec4(out_color, gl_HitTEXT);
	payload.depth += 1;
}
