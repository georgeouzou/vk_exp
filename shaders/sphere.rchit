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
hitAttributeEXT vec3 sphere_point;

void main()
{
	SpherePrimitive sph = sphere_buffer.spheres[gl_PrimitiveID];
	const vec3 aabb_max = vec3(sph.aabb_maxx, sph.aabb_maxy, sph.aabb_maxz);
	const vec3 aabb_min = vec3(sph.aabb_minx, sph.aabb_miny, sph.aabb_minz);
	const vec3 center = (aabb_max + aabb_min) / vec3(2.0);

	const vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	const vec3 hit_normal = normalize(hit_pos - center);
	//const vec3 hit_normal = normalize(sphere_point - center);
	//const vec3 hit_pos = sphere_point;
	
	const bool metallic = sph.material == 1;
	const bool emissive = sph.material == 2;
	vec3 scatter_dir;
	bool scatters;
	vec3 emissive_color = vec3(0.0);
	if (metallic) {
		vec3 reflected = reflect(gl_WorldRayDirectionEXT, hit_normal);
		scatter_dir = reflected + sph.fuzz*random_in_unit_sphere(payload.seed);;
		scatters = dot(scatter_dir, hit_normal) > 0.0;
	} else if (emissive) {
		emissive_color = sph.albedo.rgb;
		scatter_dir = vec3(0.0);
		scatters = false;
	} else { // lambertian 
		scatter_dir = random_in_hemisphere(payload.seed, hit_normal);
		scatters = true;
	}

	const vec3 attenuation = sph.albedo.rgb;

	payload.ray_dir = scatter_dir;
	payload.ray_t = gl_HitTEXT;
	payload.scatter_color = attenuation;
	payload.scatters = scatters;
	payload.emissive_color = emissive_color;
	payload.emits = emissive;
}
