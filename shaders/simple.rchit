#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct HitPayload
{
	vec4 color_dist;
	int depth;
};

struct ShadowPayload
{
	float in_shadow;
};

struct TriVertex
{
	vec4 pos;
	vec4 normal;
	vec4 tex_coord;
};

layout(buffer_reference, scalar, buffer_reference_align = 8) buffer VertexBuffer
{
	TriVertex vertices[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer IndexBuffer
{
	uvec3 indices[];
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
	VertexBuffer vertex_buffer;
	IndexBuffer index_buffer;
    vec4 part_color;
} shader_record;

layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(location = 1) rayPayloadEXT ShadowPayload shadow_payload;
hitAttributeEXT vec2 bary;

void main()
{
	vec3 barys = vec3(1.0f - bary.x - bary.y, bary.x, bary.y);

	VertexBuffer vbuf = shader_record.vertex_buffer;
	IndexBuffer ibuf = shader_record.index_buffer;

	uvec3 vidx = ibuf.indices[gl_PrimitiveID];
	TriVertex v0 = vbuf.vertices[vidx.x];
	TriVertex v1 = vbuf.vertices[vidx.y];
	TriVertex v2 = vbuf.vertices[vidx.z];

	vec2 texc = v0.tex_coord.xy * barys.x +
				v1.tex_coord.xy * barys.y +
				v2.tex_coord.xy * barys.z;

	vec3 norm = v0.normal.xyz * barys.x +
				v1.normal.xyz * barys.y +
				v2.normal.xyz * barys.z;
	norm = normalize(vec3(ubo.model * vec4(norm, 0.0)));
	
	vec3 hit_color = shader_record.part_color.rgb;

	const vec3 hit_normal = norm;
	const vec3 hit_pos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
	const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	const vec3 to_light1 = normalize(ubo.light_pos.xyz - shadow_ray_orig);

	const uint shadow_ray_flags = gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT; // | gl_RayFlagsSkipClosestHitShaderEXT;
	//shadow_payload.in_shadow = 1.0;
	traceRayEXT(scene, shadow_ray_flags, 0xFF, 1, 2, 1, shadow_ray_orig, 0.01, to_light1, 1000.0, 1);

	const float ambient = 0.1;
	const float lighting1 = shadow_payload.in_shadow > 0.0 ? ambient : max(ambient, dot(hit_normal, to_light1));
	
	vec3 out_color = lighting1 * hit_color;
	payload.color_dist = vec4(out_color, gl_HitTEXT);
	payload.depth += 1;
}
