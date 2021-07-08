#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "common.glsl"
#include "random.glsl"

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
    vec4 part_color;
} shader_record;

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;
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

void main()
{
	vec3 color = shader_record.part_color.rgb;
	const vec3 hit_normal = fetch_normal(gl_PrimitiveID);

	const vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	
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
	
	//const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	//const vec3 to_light1 = normalize(ubo.light_pos.xyz - shadow_ray_orig);
	//
	//const uint shadow_ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT; // | gl_RayFlagsSkipClosestHitShaderEXT;
	////shadow_payload.in_shadow = 1.0;
	//traceRayEXT(scene, shadow_ray_flags, 0xFF, 1, 2, 1, shadow_ray_orig, 0.01, to_light1, 1000.0, 1);
	//
	//const float ambient = 0.1;
	//const float lighting1 = shadow_payload.in_shadow > 0.0 ? ambient :  max(ambient, dot(hit_normal, to_light1));
	//vec3 out_color = lighting1 * color;

	payload.color = vec4(color, 1.0);
}
