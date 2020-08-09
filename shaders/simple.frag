#version 460
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require

layout(binding = 0) uniform CameraMatrices 
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 iview;
	mat4 iproj;
	vec4 light_pos;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D tex_sampler;
layout(set = 0, binding = 2) uniform accelerationStructureEXT scene;

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_tex_coord;
layout(location = 2) in vec4 frag_world_pos;

layout(location = 0) out vec4 out_color;

void main()
{
	const vec3 hit_normal = frag_normal;
	const vec3 hit_pos = frag_world_pos.xyz;
	const vec3 shadow_ray_orig = hit_pos + hit_normal * 0.001f;
	const vec3 to_light = normalize(ubo.light_pos.xyz-hit_pos.xyz);

	vec4 color = texture(tex_sampler, frag_tex_coord);

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, scene, 
        gl_RayFlagsTerminateOnFirstHitEXT, 0xFF, 
        shadow_ray_orig, 0.001, 
        to_light, 1000.0);

    while (rayQueryProceedEXT(ray_query)) {}

    bool in_shadow = rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
    const float ambient = 0.1;
	const float lighting = (in_shadow) ? ambient : max(ambient, dot(hit_normal, to_light));
	out_color = lighting * color; 
}
