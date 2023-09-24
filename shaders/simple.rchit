#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "common.glsl"
#include "random.glsl"

// Code for materials is based on:
// https://raytracing.github.io/books/RayTracingInOneWeekend.html

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VertexBuffer
{
	TriVertex vertices[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer IndexBuffer
{
	uvec3 indices[];
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene;

layout(set = 0, binding = 2, std140) uniform SceneUniformsBlock 
{
	SceneUniforms ubo;
};

layout(shaderRecordEXT, std430) buffer ShaderRecord
{
	VertexBuffer vertex_buffer;
	IndexBuffer index_buffer;
    PBRMaterial material;
} shader_record;

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 bary;

vec3 fetch_normal(uint primitive_id)
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);
	VertexBuffer vbuf = shader_record.vertex_buffer;
	IndexBuffer ibuf = shader_record.index_buffer;

	uvec3 vidx = ibuf.indices[primitive_id];
	TriVertex v0 = vbuf.vertices[vidx.x];
	TriVertex v1 = vbuf.vertices[vidx.y];
	TriVertex v2 = vbuf.vertices[vidx.z];

	vec3 norm = v0.normal.xyz * barys.x +
				v1.normal.xyz * barys.y +
				v2.normal.xyz * barys.z;
	norm = normalize(vec3(ubo.model * vec4(norm, 0.0)));
	return norm;
}


float reflectance(float cosine, float ref_idx)
{
	// Use Schlick's approximation for reflectance.
	float r0 = (1.0-ref_idx) / (1.0+ref_idx);
	r0 = r0*r0;
	return r0 + (1.0-r0)*pow((1-cosine), 5.0);
}

void main()
{
	const vec3 hit_normal = fetch_normal(gl_PrimitiveID);
	const PBRMaterial material = shader_record.material;

	const vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	
	const bool metallic = material.metallic > 0.3;
	const bool transparent = material.albedo.a < 1.0;
	vec3 scatter_dir;
	bool scatters;
	vec3 attenuation;
	if (transparent) {
		const float ior = material.ior;
		const float ratio = gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT ? (1.0 / ior) : ior;
		const float cos_theta = min(dot(-gl_WorldRayDirectionEXT, hit_normal), 1.0);
		const float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
		const bool cannot_refract = ratio * sin_theta > 1.0;
		const float reflectance =  reflectance(cos_theta, ratio);
		if (cannot_refract || reflectance > random_float(payload.seed)) {
			scatter_dir = reflect(gl_WorldRayDirectionEXT, hit_normal);
		} else {
			scatter_dir = refract(gl_WorldRayDirectionEXT, hit_normal, ratio);
		}
		scatters = true;
		attenuation = vec3(1.0);
	} else if (metallic) {
		vec3 reflected = reflect(gl_WorldRayDirectionEXT, hit_normal);
		scatter_dir = reflected + material.roughness*random_in_unit_sphere(payload.seed);
		scatters = dot(scatter_dir, hit_normal) > 0.0;
		attenuation = material.albedo.rgb;
	} else {
		scatter_dir = random_in_hemisphere(payload.seed, hit_normal);
		scatters = true;
		attenuation = material.albedo.rgb;
	}
	
	payload.ray_dir = scatter_dir;
	payload.ray_t = gl_HitTEXT;
	payload.scatter_color = attenuation;
	payload.scatters = scatters;
	payload.emissive_color = vec3(0.0);
	payload.emits = false;
}
