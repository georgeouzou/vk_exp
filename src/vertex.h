#ifndef VERTEX_H
#define VERTEX_H

#include <array>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>
#include <glm/gtx/hash.hpp>
#include <vulkan/vulkan.h>

struct Vertex
{
	glm::vec3 pos;
	float pad0;
	glm::vec3 normal;
	float pad1;
	glm::vec2 tex_coord;
	glm::vec2 pad2;

	bool operator == (const Vertex &other) const
	{
		return pos == other.pos && normal == other.normal && tex_coord == other.tex_coord;
	}

	template<int DIM>
	static bool compare_position(const Vertex &v0, const Vertex &v1)
	{
		static_assert(DIM < 3);
		return v0.pos[DIM] < v1.pos[DIM];
	}

	static VkVertexInputBindingDescription get_binding_description()
	{
		VkVertexInputBindingDescription bd = {};
		bd.binding = 0;
		bd.stride = sizeof(Vertex);
		bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bd;
	}

	static std::array<VkVertexInputAttributeDescription, 3>
		get_attribute_descriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> ad;
		ad[0].binding = 0;
		ad[0].location = 0;
		ad[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		ad[0].offset = offsetof(Vertex, pos);
		ad[1].binding = 0;
		ad[1].location = 1;
		ad[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		ad[1].offset = offsetof(Vertex, normal);
		ad[2].binding = 0;
		ad[2].location = 2;
		ad[2].format = VK_FORMAT_R32G32_SFLOAT;
		ad[2].offset = offsetof(Vertex, tex_coord);

		return ad;
	}
};

static_assert(sizeof(Vertex) % 8 == 0 && "We have chosen vertices to have an alignment of 8");

// implement has specialization for vertex
namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^
					(hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^
					(hash<glm::vec2>()(vertex.tex_coord) << 1);
		}
	};
}

#endif

