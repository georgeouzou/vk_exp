#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <glm/mat4x4.hpp>
#include "vertex.h"
#include "materials.hpp"

struct ModelPart
{
    uint32_t vertex_offset;
    uint32_t vertex_count;
    uint32_t index_offset;
    uint32_t index_count;
	materials::PBRMaterial pbr_material;
};

class Model
{
public:
	Model();

	inline const std::vector<Vertex>& get_vertices() const { return m_vertices; }
	inline const std::vector<uint32_t>& get_indices() const { return m_indices; }
	inline const std::vector<ModelPart>& get_parts() const { return m_parts; }
	inline glm::mat4 get_transformation() const { return m_transformation; }

private:
	void load_gltf(const char *filepath);
	void load_obj(const char *filepath, const char *basedir);

private:
	std::vector<Vertex> m_vertices;
	std::vector<uint32_t> m_indices;
	std::vector<ModelPart> m_parts;
	glm::mat4 m_transformation { 1.0f };
};

#endif // !MODEL_H
