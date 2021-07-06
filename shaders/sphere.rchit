#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct HitPayload
{
	vec4 color_dist;
	float depth;
};

struct ShadowPayload
{
	float in_shadow;
};

struct SpherePrimitive
{
	vec4 albedo;
	float aabb_minx;
	float aabb_miny;
	float aabb_minz;
	float aabb_maxx;
	float aabb_maxy;
	float aabb_maxz;
	int material;
	float fuzz;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer SphereBuffer
{
	SpherePrimitive spheres[];
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

layout(shaderRecordEXT, std430) buffer ShaderRecord
{
	SphereBuffer sphere_buffer;
} shader_record;

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;
hitAttributeEXT vec3 sphere_point;

void main()
{
	SpherePrimitive sph = shader_record.sphere_buffer.spheres[gl_PrimitiveID];
	const vec3 aabb_max = vec3(sph.aabb_maxx, sph.aabb_maxy, sph.aabb_maxz);
	const vec3 aabb_min = vec3(sph.aabb_minx, sph.aabb_miny, sph.aabb_minz);
	const vec3 center = (aabb_max + aabb_min) / vec3(2.0);
	const float radius = (aabb_max.x - aabb_min.x) / 2.0;
	const bool metallic = sph.material == 1;

	vec3 norm = normalize(sphere_point - center);
	vec3 refl_dir = reflect(normalize(gl_WorldRayDirectionEXT), norm);
	const vec3 hit_pos = sphere_point;
	
	const vec3 refl_orig = hit_pos + norm * 0.001;

	const uint ray_flags = gl_RayFlagsOpaqueEXT;

	vec3 color = sph.albedo.rgb;
	
	//if (metallic && payload.depth < 2) {
	//	traceRayEXT(scene, ray_flags, 0xFF, 0, 2, 0, refl_orig, 0.001, refl_dir, 1000.0, 0);
	//	vec3 in_color = payload.color_dist.rgb;	
	//	color = color * in_color;
	//}

	const vec3 to_light1 = normalize(ubo.light_pos.xyz - refl_orig);
	
	const uint shadow_ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT; // | gl_RayFlagsSkipClosestHitShaderEXT;
	
	//shadow_payload.in_shadow = 1.0;
	traceRayEXT(scene, shadow_ray_flags, 0xFF, 1, 2, 1, refl_orig, 0.01, to_light1, 1000.0, 1);

	const float ambient = 0.1;
	const float lighting1 = shadow_payload.in_shadow > 0.0 ? ambient : max(ambient, dot(norm, to_light1));
	
	vec3 out_color = lighting1 * color;
	payload.color_dist = vec4(out_color, gl_HitTEXT);
	payload.depth += 1;
}
