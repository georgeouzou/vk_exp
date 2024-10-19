#include "model.h"

#include <string>
#include <cstdio>
#include <cinttypes>
#include <stdexcept>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION 
#include <tiny_gltf.h>
#include <tiny_obj_loader.h>

Model::Model()
{
	load_gltf("resources/Sponza/glTF/Sponza.gltf");
	//load_obj("resources/bmw.obj", "resources");
}

void Model::load_gltf(const char *filepath)
{
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	const std::string filename = filepath;
	const bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename.c_str());

	if (!warn.empty()) {
		printf("GLTF loader warning: %s\n", warn.c_str());
	}
	if (!err.empty()) {
		printf("GLTF loader error: %s\n", err.c_str());
	}
	if (!ret) {
		printf("GLTF loader error: Failed to load %s\n", filename.c_str());
	}

	const tinygltf::Scene &scene = model.scenes[model.defaultScene];
	for (size_t i = 0; i < scene.nodes.size(); ++i) {
		assert(scene.nodes[i] >= 0 && scene.nodes[i] < model.nodes.size());
		const tinygltf::Node& node = model.nodes[scene.nodes[i]];
		const int mesh_idx = node.mesh;
		const size_t children = node.children.size();
		assert(children == 0); //TODO

		const glm::mat4 scale = (node.scale.size() == 3) ?
			glm::scale(glm::mat4(1.0f), glm::vec3(node.scale[0], node.scale[1], node.scale[2])) :
			glm::mat4(1.0f);
		const glm::mat4 rotation = (node.rotation.size() == 4) ?
			glm::mat4_cast(glm::quat(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2])) :
			glm::mat4(1.0f);
		const glm::mat4 translation = (node.translation.size() == 3) ?
			glm::translate(glm::mat4(1.0f), glm::vec3(node.translation[0], node.translation[1], node.translation[2])) :
			glm::mat4(1.0f);

		m_transformation = translation * rotation * scale;

		if (mesh_idx >= 0) {
			const tinygltf::Mesh& mesh = model.meshes[mesh_idx];

			for (size_t prim_idx = 0; prim_idx < mesh.primitives.size(); ++prim_idx) {
				const tinygltf::Primitive& primitive = mesh.primitives[prim_idx];

				const uint32_t index_offset = m_indices.size();
				const uint32_t vertex_offset = m_vertices.size();

				assert(primitive.mode == TINYGLTF_MODE_TRIANGLES);
				assert(primitive.indices >= 0);
				{
					// indices
					const tinygltf::Accessor& index_accessor = model.accessors[primitive.indices];

					const tinygltf::BufferView& bv = model.bufferViews[index_accessor.bufferView];
					const size_t offset = index_accessor.byteOffset + bv.byteOffset;
					std::vector<uint32_t> part_indices(index_accessor.count);
					const uint8_t* src = model.buffers[bv.buffer].data.data() + offset;
					if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
						const uint16_t* src16 = reinterpret_cast<const uint16_t*>(src);
						std::transform(src16, src16 + index_accessor.count, part_indices.begin(),
							[](const uint16_t idx) {
								return uint32_t(idx);
							});
					} else {
						assert(index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
						std::memcpy(part_indices.data(), src, sizeof(uint32_t) * part_indices.size());
					}
					m_indices.insert(m_indices.end(), part_indices.begin(), part_indices.end());
				}

				{
					// positions, normals, texcoords
					const auto position_attr_it = primitive.attributes.find("POSITION");
					const auto normal_attr_it = primitive.attributes.find("NORMAL");
					const auto texcoord_attr_it = primitive.attributes.find("TEXCOORD_0");
					assert(position_attr_it != primitive.attributes.end());
					assert(normal_attr_it != primitive.attributes.end());
					assert(texcoord_attr_it != primitive.attributes.end());

					const tinygltf::Accessor& position_accessor = model.accessors[position_attr_it->second];
					const tinygltf::Accessor& normal_accessor = model.accessors[normal_attr_it->second];
					const tinygltf::Accessor& texcoord_accessor = model.accessors[texcoord_attr_it->second];
					assert(position_accessor.count == normal_accessor.count);
					assert(position_accessor.count == texcoord_accessor.count);
					assert(texcoord_accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

					std::vector<Vertex> part_vertices(position_accessor.count);
					{
						const tinygltf::BufferView& bv = model.bufferViews[position_accessor.bufferView];
						const size_t offset = position_accessor.byteOffset + bv.byteOffset;
						const size_t stride = position_accessor.ByteStride(bv);
						const uint8_t* src = model.buffers[bv.buffer].data.data() + offset;
						for (auto& vtx : part_vertices) {
							std::memcpy(glm::value_ptr(vtx.pos), src, sizeof(glm::vec3));
							src += stride;
						}
					}
					{
						const tinygltf::BufferView& bv = model.bufferViews[normal_accessor.bufferView];
						const size_t offset = normal_accessor.byteOffset + bv.byteOffset;
						const size_t stride = normal_accessor.ByteStride(bv);
						const uint8_t* src = model.buffers[bv.buffer].data.data() + offset;
						for (auto& vtx : part_vertices) {
							std::memcpy(glm::value_ptr(vtx.normal), src, sizeof(glm::vec3));
							src += stride;
						}
					}
					{
						const tinygltf::BufferView& bv = model.bufferViews[texcoord_accessor.bufferView];
						const size_t offset = texcoord_accessor.byteOffset + bv.byteOffset;
						const size_t stride = texcoord_accessor.ByteStride(bv);
						const uint8_t* src = model.buffers[bv.buffer].data.data() + offset;
						for (auto& vtx : part_vertices) {
							std::memcpy(glm::value_ptr(vtx.tex_coord), src, sizeof(glm::vec2));
							src += stride;
						}
					}
					m_vertices.insert(m_vertices.end(), part_vertices.begin(), part_vertices.end());
				}

				ModelPart part;
				part.index_offset = index_offset;
				part.vertex_offset = vertex_offset;
				part.index_count = m_indices.size() - index_offset;
				part.vertex_count = m_vertices.size() - vertex_offset;
				part.pbr_material.albedo = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
				part.pbr_material.metallic = 0.0f;
				part.pbr_material.roughness = 1.0f;
				part.pbr_material.ior = 1.0f;

				m_parts.push_back(part);
			}
		}
	}
}

void Model::load_obj(const char* filepath, const char *basedir)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> parts;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &parts, &materials, &warn, &err, filepath, basedir)) {
		throw std::runtime_error(warn + err);
	}

	for (const auto& part : parts) {
		std::unordered_map<Vertex, uint32_t> unique_vtx = {};
		std::vector<Vertex> part_vertices;
		std::vector<uint32_t> part_indices;
		for (const auto& index : part.mesh.indices) {
			Vertex vertex = {};
			if (index.vertex_index < 0 || index.texcoord_index < 0 || index.normal_index < 0) continue;
			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2],
			};
			vertex.tex_coord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};
			vertex.normal = {
				attrib.normals[3 * index.normal_index + 0],
				attrib.normals[3 * index.normal_index + 1],
				attrib.normals[3 * index.normal_index + 2]
			};
			if (unique_vtx.count(vertex) == 0) {
				unique_vtx[vertex] = uint32_t(part_vertices.size());
				part_vertices.push_back(vertex);
			}
			part_indices.push_back(unique_vtx[vertex]);
		}
		// append part data to model data
		if (part_indices.size() == 0) {
			assert(part_vertices.size() == 0);
			continue;
		}

		uint32_t vertex_offset = m_vertices.size();
		uint32_t index_offset = m_indices.size();
		m_vertices.insert(m_vertices.end(), part_vertices.begin(), part_vertices.end());
		m_indices.insert(m_indices.end(), part_indices.begin(), part_indices.end());

		ModelPart part_info = {};
		part_info.vertex_offset = vertex_offset;
		part_info.vertex_count = part_vertices.size();
		part_info.index_offset = index_offset;
		part_info.index_count = part_indices.size();

		// set material
		tinyobj::material_t tmat = materials[part.mesh.material_ids[0]];
		materials::MTLMaterial mtl;
		mtl.diffuse_color = glm::make_vec3(&tmat.diffuse[0]);
		mtl.ns = tmat.shininess;
		mtl.specular_color = glm::make_vec3(&tmat.specular[0]);

		part_info.pbr_material = materials::convert_mtl_to_pbr(mtl);
		part_info.pbr_material.albedo.a = tmat.dissolve;
		part_info.pbr_material.ior = tmat.ior;
		m_parts.push_back(part_info);
		printf("Add part %s {v0 %u, vc %u, i0 %u, ic %u}\t material [albedo {%.2f, %.2f, %.2f, %.2f}, metallic %.2f, roughness %.2f\n",
			part.name.c_str(),
			part_info.vertex_offset, part_info.vertex_count, part_info.index_offset, part_info.index_count,
			part_info.pbr_material.albedo.r, part_info.pbr_material.albedo.g,
			part_info.pbr_material.albedo.b, part_info.pbr_material.albedo.a,
			part_info.pbr_material.metallic, part_info.pbr_material.roughness);
	}
	fprintf(stdout, "Loaded model part: num vertices %" PRIu64 ", num indices %" PRIu64 "\n",
		m_vertices.size(),
		m_indices.size());

	auto [vxmin, vxmax] = std::minmax_element(m_vertices.begin(), m_vertices.end(), Vertex::compare_position<0>);
	auto [vymin, vymax] = std::minmax_element(m_vertices.begin(), m_vertices.end(), Vertex::compare_position<1>);
	auto [vzmin, vzmax] = std::minmax_element(m_vertices.begin(), m_vertices.end(), Vertex::compare_position<2>);
	glm::vec3 min_coord(vxmin->pos.x, vymin->pos.y, vzmin->pos.z);
	glm::vec3 max_coord(vxmax->pos.x, vymax->pos.y, vzmax->pos.z);
	glm::vec3 diff_coord = max_coord - min_coord;
	float scale = std::min({ diff_coord.x, diff_coord.y, diff_coord.z });
	glm::mat4 model_scale = glm::scale(glm::mat4(1.0), glm::vec3(1.0f / scale));

	m_transformation = model_scale;// glm::mat4(1.0f);
}









