#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform CameraMatrices 
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 iview;
	mat4 iproj;
} ubo;


layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_tex_coord;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec2 frag_tex_coord;

void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(in_position, 1.0);
	frag_color = in_color;
	frag_tex_coord = in_tex_coord;
}