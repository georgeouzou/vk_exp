#version 460
#extension GL_NV_ray_tracing : require

struct HitPayload
{
	vec4 color_dist;
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

layout(std430, binding = 6) readonly buffer SpherePrimitives
{
	SpherePrimitive spheres[];
};

layout(location = 0) rayPayloadInNV HitPayload payload;

hitAttributeNV vec3 sphere_point;

void main()
{
	SpherePrimitive sph = spheres[gl_PrimitiveID];
	payload.color_dist = vec4(sph.albedo.rgb, 1.0);
}