#ifndef COMMON_H_GLSL
#define COMMON_H_GLSL

struct HitPayload
{
	uint seed;
	vec3 ray_dir;
	uint color;
	vec3 ray_orig;
	bool scatters;
	bool hit;
};

struct ShadowPayload
{
	float in_shadow;
};

struct SceneUniforms
{
	mat4 model;
	mat4 view;
	mat4 proj;
	mat4 iview;
	mat4 iproj;
	vec4 light_pos;
	uint samples_accum;
	uint pad0;
	uint pad1;
	uint pad2;
};

struct SpherePrimitive
{
	vec4 albedo;
	float aabb_minx;
	float aabb_miny;
	float aabb_minz;
	float aabb_maxx;
	float aabb_maxy;
	float aabb_maxz;
	int material;
	float fuzz;
};

struct TriVertex
{
	vec4 pos;
	vec4 normal;
	vec4 tex_coord;
};

struct PBRMaterial 
{
	vec4 albedo;
	float metallic;
	float roughness;
	float ior;
	float pad1;
};

#endif //COMMON_H_GLSL