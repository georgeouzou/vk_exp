#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform CameraMatrices 
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 iview;
	mat4 iproj;
	vec4 light_pos;
} ubo;


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_tex_coord;

layout(location = 0) 
out VertexOut
{
	vec3 wnormal;
	vec2 tex_coord;
	vec4 wpos;
} vs_out;


void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_position, 1.0);
	vs_out.wnormal = (ubo.model * vec4(in_normal, 0.0)).xyz;
	vs_out.tex_coord = in_tex_coord;
    vs_out.wpos = ubo.model * vec4(in_position, 1.0);
}
