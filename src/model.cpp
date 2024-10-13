#include "model.h"

#include <string>
#include <cstdio>
#include <glm/gtc/type_ptr.hpp>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION 
#include <tiny_gltf.h>

Model::Model()
{
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	const std::string filename = "resources/Sponza/glTF/Sponza.gltf";
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
		const int mesh_idx = model.nodes[scene.nodes[i]].mesh;
		const size_t children = model.nodes[scene.nodes[i]].children.size();
		assert(children == 0); //TODO
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









