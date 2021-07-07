
struct HitPayload
{
	vec4 color_dist;
	int depth;
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