#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "common.glsl"
#include "random.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;

layout(set = 0, binding = 2, std140) uniform SceneUniformsBlock 
{
	SceneUniforms ubo;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer SphereBuffer
{
	SpherePrimitive spheres[];
};

layout(shaderRecordEXT, std430) buffer ShaderRecord
{
	SphereBuffer sphere_buffer;
};

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;
hitAttributeEXT vec3 sphere_point;

void main()
{
	

	SpherePrimitive sph = sphere_buffer.spheres[gl_PrimitiveID];
	const vec3 aabb_max = vec3(sph.aabb_maxx, sph.aabb_maxy, sph.aabb_maxz);
	const vec3 aabb_min = vec3(sph.aabb_minx, sph.aabb_miny, sph.aabb_minz);
	const vec3 center = (aabb_max + aabb_min) / vec3(2.0);
	const float radius = (aabb_max.x - aabb_min.x) / 2.0;
	const bool metallic = sph.material == 1;

	const vec3 hit_normal = normalize(sphere_point - center);
	const vec3 hit_pos = sphere_point;
	
	vec3 refl_dir = reflect(normalize(gl_WorldRayDirectionEXT), hit_normal);
	const vec3 refl_orig = hit_pos;

	vec3 color = sph.albedo.rgb;
	vec3 scatter_dir = random_in_hemisphere(payload.seed, hit_normal);

	if (payload.depth < 16) {
		const uint ray_flags = gl_RayFlagsOpaqueEXT;
		payload.depth += 1;
		traceRayEXT(scene, ray_flags, 0xFF, 0, 2, 0, hit_pos, 0.001, scatter_dir, 100.0, 0);
		vec3 in_color = payload.color.rgb;
		color = in_color * color;
	} else {
		color = vec3(0.1); // stop accumulating light
	}

	//const vec3 to_light1 = normalize(ubo.light_pos.xyz - hit_pos);
	//const uint shadow_ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT; // | gl_RayFlagsSkipClosestHitShaderEXT;
	////shadow_payload.in_shadow = 1.0;
	//traceRayEXT(scene, shadow_ray_flags, 0xFF, 1, 2, 1, hit_pos, 0.01, to_light1, 1000.0, 1);
	//const float ambient = 0.1;
	//const float lighting1 = shadow_payload.in_shadow > 0.0 ? ambient : max(ambient, dot(hit_normal, to_light1));
	//
	//vec3 out_color = lighting1 * color;

	payload.color = vec4(color, 1.0);
}
