#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

#include "common.glsl"

layout(push_constant) uniform PushConstantsBlock
{
	PBRMaterial material;
} pc;

layout(set = 0, binding = 0, std140) uniform SceneUniformsBlock 
{
	SceneUniforms ubo;
};

layout(set = 0, binding = 1) uniform accelerationStructureEXT scene;

layout(location = 0) 
in VertexOut
{
	vec3 wnormal;
	vec2 tex_coord;
	vec4 wpos;
} fs_in;

layout(location = 0) out vec4 out_color;

void main()
{
	const vec3 hit_normal = normalize(fs_in.wnormal);
	const vec3 hit_pos = fs_in.wpos.xyz;
	const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	const vec3 to_light = normalize(ubo.light_pos.xyz-hit_pos.xyz);

	vec4 color = pc.material.albedo;

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, scene, 
        gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 
        shadow_ray_orig, 0.001, 
        to_light, 1000.0);

    while (rayQueryProceedEXT(ray_query)) {}

    bool in_shadow = rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
    //bool in_shadow = false;
	const float ambient = 0.1;
	const float lighting = (in_shadow) ? ambient : max(ambient, dot(hit_normal, to_light));
	color = lighting * color; 

	// gamma correct, for gamma = 2.0
	out_color = sqrt(color);
}
