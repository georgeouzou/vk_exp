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

	const vec3 hit_normal = normalize(sphere_point - center);
	const vec3 hit_pos = sphere_point;
	
	const bool metallic = sph.material == 1;
	const bool emissive = sph.material == 2;
	vec3 scatter_dir;
	bool scatter;
	vec3 emitted = vec3(0.0);
	if (metallic) {
		vec3 reflected = reflect(gl_WorldRayDirectionEXT, hit_normal);
		scatter_dir = reflected + sph.fuzz*random_in_unit_sphere(payload.seed);;
		scatter = dot(scatter_dir, hit_normal) > 0.0;
	} else if (emissive) {
		emitted = sph.albedo.rgb;
		scatter_dir = vec3(0.0);
		scatter = false;
	} else { // lambertian 
		scatter_dir = random_in_hemisphere(payload.seed, hit_normal);
		scatter = true;
	}

	const vec3 attenuation = sph.albedo.rgb;

	const uint ray_flags = gl_RayFlagsOpaqueEXT;
	payload.depth += 1;
	bool can_recurse = payload.depth < 16; // or 15 ???
	uint mask = (can_recurse && scatter) ? 0xFF : 0;
	
	traceRayEXT(scene, ray_flags, mask, 0, 2, 0, hit_pos, 0.001, scatter_dir, 100.0, 0);
	
	vec3 accum = scatter ? (attenuation * payload.color.rgb) : vec3(0.0);
	vec3 color = emitted + accum;
	
	if (!can_recurse) { // check if all was invalid due to recursion depth
		color = vec3(0.0); // stop accumulating light
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
