#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <cinttypes>
#include <functional>
#include <vector>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>
#include <random>
#include <cassert>

#include <volk.h>
#include <shaderc/shaderc.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // projection matrix depth range 0-1
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>

#include <GLFW/glfw3.h>
#include <tiny_obj_loader.h>

#include "vma.h"
#include "stb_image.h"
#include "orbit_camera.h"
#include "shader_dir.h"
#include "materials.hpp"

const int MAX_FRAMES_IN_FLIGHT = 3;
//#define ENABLE_VALIDATION_LAYERS

struct ShaderGroupHandle
{
	uint64_t i0;
	uint64_t i1;
	uint64_t i2;
	uint64_t i3;
};

struct VmaBufferAllocation
{
	VmaAllocation alloc{ VK_NULL_HANDLE };
	VkBuffer buffer{ VK_NULL_HANDLE };
};

struct VmaImageAllocation
{
	VmaAllocation alloc{ VK_NULL_HANDLE };
	VkImage image{ VK_NULL_HANDLE };
};

struct ASBuffers
{
	VkAccelerationStructureKHR structure{ VK_NULL_HANDLE };
	VmaBufferAllocation structure_buffer;
	VmaBufferAllocation scratch_buffer;
	VmaBufferAllocation instances_buffer;
	
	void destroy(VkDevice device, VmaAllocator allocator)
	{
		if (structure) vkDestroyAccelerationStructureKHR(device, structure, nullptr);
		vmaDestroyBuffer(allocator, structure_buffer.buffer, structure_buffer.alloc);
		vmaDestroyBuffer(allocator, scratch_buffer.buffer, scratch_buffer.alloc);
		vmaDestroyBuffer(allocator, instances_buffer.buffer, instances_buffer.alloc);
		std::memset(this, 0, sizeof(ASBuffers));
	}
};


struct QueueFamilyIndices
{
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;
	std::optional<uint32_t> transfer_family;
	
	bool is_complete()
	{
		return graphics_family.has_value() && present_family.has_value()
			&& transfer_family.has_value();
	}
};

struct SwapchainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities{};
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

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

struct SpherePrimitive
{
	glm::vec4 albedo;
	VkAabbPositionsKHR bbox;
	materials::MaterialType material; //int
	float fuzz;
};

static_assert(sizeof(SpherePrimitive) % 8 == 0 && "We have chosen SpherePrimitives to have an alignment of 8");

struct SceneUniforms
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 iview;
	glm::mat4 iproj;
	glm::vec4 light_pos;
	uint32_t samples_accum;
	uint32_t pad0;
	uint32_t pad1;
	uint32_t pad2;
};

struct ModelPart
{
    uint32_t vertex_offset;
    uint32_t vertex_count;
    uint32_t index_offset;
    uint32_t index_count;
	materials::PBRMaterial pbr_material;
};

struct SBTRecordHitMesh
{
	ShaderGroupHandle shader;
	VkDeviceAddress vertices_ref;
	VkDeviceAddress indices_ref;
	materials::PBRMaterial pbr_material;
};

struct SBTRecordHitSphere
{
	ShaderGroupHandle shader;
	VkDeviceAddress spheres_ref;
};

constexpr size_t get_sbt_hit_record_size()
{
	const size_t sz = std::max({sizeof(SBTRecordHitMesh), sizeof(SBTRecordHitSphere), sizeof(ShaderGroupHandle)});
	// round up to 64
	return ((sz + 63) / 64) * 64;
}

constexpr size_t get_sbt_miss_record_size()
{
	const size_t sz = sizeof(ShaderGroupHandle);
	// round up to 64
	return ((sz + 63) / 64) * 64;
}

constexpr size_t get_sbt_raygen_record_size()
{
	const size_t sz = sizeof(ShaderGroupHandle);
	// round up to 64
	return ((sz + 63) / 64) * 64;
}

class BaseApplication
{
public:
	BaseApplication();
	~BaseApplication();
	void run();
	void on_window_resized() { m_window_resized = true; }
	void on_accumulated_samples_reset() { m_samples_accumulated = 0; };
	void on_toggle_raytracing() { m_raytraced = !m_raytraced; }
	
	OrbitCamera &camera() { return m_camera; }

private:
	void init_window();
	void init_vulkan();
	void main_loop();
	void cleanup();

	void create_instance();
	bool check_validation_layer_support() const;
	std::vector<const char*> get_required_instance_extensions() const;

	void setup_debug_callback();
	void destroy_debug_callback();

	void pick_gpu();
	bool check_device_extension_support(VkPhysicalDevice gpu) const;
	bool is_gpu_suitable(VkPhysicalDevice gpu) const;
	QueueFamilyIndices find_queue_families(VkPhysicalDevice gpu) const;

	void create_logical_device();

	void create_allocator();

	void create_surface();
	
	SwapchainSupportDetails query_swapchain_support(VkPhysicalDevice gpu) const;
	VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR> &formats) const;
	VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR> modes) const;
	VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities) const;
	void create_swapchain();
	void create_image_views();

	void create_render_pass();
	void create_descriptor_set_layout();
	void create_graphics_pipeline();

	VkShaderModule create_shader_module(const std::string& file_name, shaderc_shader_kind shader_kind, const std::vector<char>& code) const;

	void create_framebuffers();

	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
	
	void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
					   VkMemoryPropertyFlags props, VmaBufferAllocation &buffer);
	void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
	
	void create_image(uint32_t width, uint32_t height, VkFormat format,
		VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags props,
		VmaImageAllocation& img);
	
	VkCommandBuffer begin_single_time_commands(VkQueue queue, VkCommandPool cmd_pool);
	void end_single_time_commands(VkQueue queue, VkCommandPool cmd_pool, VkCommandBuffer cmd_buffer);

	void transition_image_layout(VkImage img, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout);
	void copy_buffer_to_image(VkBuffer buffer, VkImage img, uint32_t width, uint32_t height);

	void load_model();
	void create_spheres();

	void create_vertex_buffer();
	void create_index_buffer();
	void create_uniform_buffers();

	void create_sphere_buffer();

	void create_bottom_acceleration_structure();
	void create_bottom_acceleration_structure_spheres();
	void create_top_acceleration_structure();
	void create_raytracing_pipeline_layout();
	void create_raytracing_pipeline();

	void create_rt_image();
	void create_descriptor_pool();
	void create_descriptor_sets();
	void create_rt_descriptor_sets();
	void create_shader_binding_table();

	VkFormat find_supported_format(const std::vector<VkFormat> &candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const;
	VkFormat find_supported_depth_format() const;
	void create_depth_resources();

	void create_command_pools();
	void create_command_buffers();
	void create_rt_command_buffers();
	
	void create_sync_objects();

	void update_uniform_buffer(uint32_t idx);
	void draw_frame();

	void cleanup_swapchain();
	void recreate_swapchain();

private:
	GLFWwindow *m_window{ nullptr };
	uint32_t m_width{ 1920 };
	uint32_t m_height{ 1080 };
	bool m_raytraced{ true };
	OrbitCamera m_camera;

	std::vector<const char*> m_validation_layers;
	bool m_enable_validation_layers;

	std::vector<const char*> m_device_extensions;
	size_t m_current_frame_idx{ 0 };
	bool m_window_resized{ false };

	shaderc::Compiler m_shader_compiler;

	VkInstance m_instance{ VK_NULL_HANDLE };
	VkDebugUtilsMessengerEXT m_debug_callback{ VK_NULL_HANDLE };
	VkPhysicalDevice m_gpu{ VK_NULL_HANDLE };

	VkDevice m_device{ VK_NULL_HANDLE };
	VkQueue m_graphics_queue{ VK_NULL_HANDLE };
	VkQueue m_present_queue{ VK_NULL_HANDLE };
	VkQueue m_transfer_queue{ VK_NULL_HANDLE };
	
	VmaAllocator m_allocator{ VK_NULL_HANDLE };

	VkSurfaceKHR m_surface{ VK_NULL_HANDLE };

	VkSwapchainKHR m_swapchain{ VK_NULL_HANDLE };
	std::vector<VkImage> m_swapchain_images;
	VkFormat m_swapchain_img_format;
	VkExtent2D m_swapchain_extent;
	std::vector<VkImageView> m_swapchain_img_views;
	std::vector<VkFramebuffer> m_swapchain_fbs;
	
	VmaImageAllocation m_depth_img;
	VkImageView m_depth_img_view{ VK_NULL_HANDLE };

	VkRenderPass m_render_pass{ VK_NULL_HANDLE };
	VkDescriptorSetLayout m_descriptor_set_layout{ VK_NULL_HANDLE };
	VkPipelineLayout m_pipeline_layout{ VK_NULL_HANDLE };
	VkPipeline m_graphics_pipeline{ VK_NULL_HANDLE };
	
	VkDescriptorSetLayout m_rt_descriptor_set_layout{ VK_NULL_HANDLE };
	VkPipelineLayout m_rt_pipeline_layout {VK_NULL_HANDLE};
	VkPipeline m_rt_pipeline{ VK_NULL_HANDLE };
	
	glm::mat4 m_model_tranformation;
	std::vector<Vertex> m_model_vertices;
	std::vector<uint32_t> m_model_indices;
    std::vector<ModelPart> m_model_parts;
	
	std::vector<SpherePrimitive> m_sphere_primitives;

	VmaBufferAllocation m_vertex_buffer;
	VmaBufferAllocation m_index_buffer;
	VmaBufferAllocation m_sphere_buffer;
	
	ASBuffers m_bottom_as_spheres;
	ASBuffers m_bottom_as;
	ASBuffers m_top_as;
	VmaImageAllocation m_rt_img;
	VkImageView m_rt_img_view{ VK_NULL_HANDLE };
	VmaBufferAllocation m_rt_sbt;
	VkDeviceAddress m_rt_sbt_address;
	
	std::vector<VmaBufferAllocation> m_uni_buffers;

	VkDescriptorPool m_desc_pool{ VK_NULL_HANDLE };
	std::vector<VkDescriptorSet> m_desc_sets;
	std::vector<VkDescriptorSet> m_rt_desc_sets;

	VkCommandPool m_graphics_cmd_pool{ VK_NULL_HANDLE };
	VkCommandPool m_transfer_cmd_pool{ VK_NULL_HANDLE };
	std::vector<VkCommandBuffer> m_cmd_buffers;
	std::vector<VkCommandBuffer> m_rt_cmd_buffers;
	
	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_sem_img_available{ VK_NULL_HANDLE };
	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_sem_render_finished{ VK_NULL_HANDLE };
	std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_fen_flight{ VK_NULL_HANDLE };

	uint32_t m_samples_accumulated{ 0 };
};

static std::vector<char> read_file(const std::string &filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + filename);
	}
	size_t file_size = (size_t)file.tellg();
	std::vector<char> buffer(file_size);
	file.seekg(0);
	file.read(buffer.data(), file_size);
	file.close();
	return buffer;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	} else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
		app->on_toggle_raytracing();
		app->on_accumulated_samples_reset();
	}
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	if (action == GLFW_PRESS) {
		auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);
		app->camera().set_mouse_position(-int(xpos), -int(ypos));
		app->on_accumulated_samples_reset();
	}
}

static void mouse_move_callback(GLFWwindow *window, double xpos, double ypos)
{
	MouseState ms;
	ms.left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
	ms.right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
	ms.middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;

	if (!ms.left && !ms.right && !ms.middle) return;

	auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
	app->camera().mouse_move(-int(xpos), -int(ypos), ms);
	app->on_accumulated_samples_reset();
}

static void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	(void)xoffset; // unused
	auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
	app->camera().mouse_scroll(float(yoffset));
	app->on_accumulated_samples_reset();
}

static void framebuffer_resize_callback(GLFWwindow *window, int width, int height)
{
	auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
	app->on_window_resized();
	app->camera().set_window_size(width, height);
	app->on_accumulated_samples_reset();
}

namespace vk_helpers
{
static VkImageView create_image_view_2d(const VkDevice device, const VkImage img, VkFormat format, VkImageAspectFlags aspect_flags)
{
	VkImageViewCreateInfo vi = {};
	vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	vi.image = img;
	vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
	vi.format = format;
	vi.subresourceRange.aspectMask = aspect_flags;
	vi.subresourceRange.baseMipLevel = 0;
	vi.subresourceRange.levelCount = 1;
	vi.subresourceRange.baseArrayLayer = 0;
	vi.subresourceRange.layerCount = 1;

	VkImageView view = VK_NULL_HANDLE;

	auto res = vkCreateImageView(device, &vi, nullptr, &view);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create texture image view");
	}
	return view;
}

bool format_has_stencil_component(VkFormat format)
{
	return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR get_raytracing_properties(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceProperties2 props = {};
	props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_props = {};
	rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	props.pNext = &rt_props;
	vkGetPhysicalDeviceProperties2(gpu, &props);
	return rt_props;
}

void image_barrier(VkCommandBuffer cmd_buffer,
	VkImage image,
	VkImageSubresourceRange& subresource_range,
	VkPipelineStageFlags2KHR src_stage_mask,
	VkAccessFlags2KHR src_access_mask,
	VkImageLayout old_layout,
	VkPipelineStageFlags2KHR dst_stage_mask,
	VkAccessFlags2KHR dst_access_mask,
	VkImageLayout new_layout)
{
	VkImageMemoryBarrier2KHR b = {};
	b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR;
	b.srcStageMask = src_stage_mask;
	b.srcAccessMask = src_access_mask;
	b.dstStageMask = dst_stage_mask;
	b.dstAccessMask = dst_access_mask;
	b.oldLayout = old_layout;
	b.newLayout = new_layout;
	b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	b.image = image;
	b.subresourceRange = subresource_range;

	VkDependencyInfoKHR dep = {};
	dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR;
	dep.imageMemoryBarrierCount = 1;
	dep.pImageMemoryBarriers = &b;

	vkCmdPipelineBarrier2KHR(cmd_buffer, &dep);
}

VkDeviceAddress get_buffer_address(VkDevice device, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo bdai = {};
	bdai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bdai.buffer = buffer;
	auto address = vkGetBufferDeviceAddress(device, &bdai);
	return address;
}

VkDeviceAddress get_acceleration_structure_address(VkDevice device, VkAccelerationStructureKHR structure)
{
	VkAccelerationStructureDeviceAddressInfoKHR dai = {};
	dai.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	dai.accelerationStructure = structure;
	return vkGetAccelerationStructureDeviceAddressKHR(device, &dai);
}

};

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
	VkDebugUtilsMessageTypeFlagsEXT message_type,
	const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
	void *p_user_data)
{
	fprintf(stderr, "validation layer: %s\n", p_callback_data->pMessage);
	return VK_FALSE;
}

void BaseApplication::run()
{
	init_window();
	init_vulkan();
	main_loop();
}

BaseApplication::BaseApplication()
{
	m_validation_layers.push_back("VK_LAYER_KHRONOS_validation");
	m_validation_layers.push_back("VK_LAYER_LUNARG_monitor");
#if !defined(ENABLE_VALIDATION_LAYERS)
	m_enable_validation_layers = false;
#else 
	m_enable_validation_layers = true;
#endif

	m_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
	m_device_extensions.push_back(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
}

BaseApplication::~BaseApplication()
{
	cleanup();
}

void BaseApplication::init_window()
{
	auto ok = glfwInit();
	if (!ok) {
		throw std::runtime_error("could not initialize glfw lib");
	}
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_window = glfwCreateWindow(m_width, m_height, "Vulkan Raytracing", NULL, NULL);
	if (!m_window) {
		throw std::runtime_error("could not create glfw window");
	}

	glfwSetKeyCallback(m_window, key_callback);
	glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_callback);
	glfwSetMouseButtonCallback(m_window, mouse_button_callback);
	glfwSetCursorPosCallback(m_window, mouse_move_callback);
	glfwSetScrollCallback(m_window, mouse_scroll_callback);
	glfwSetWindowUserPointer(m_window, this);

	m_camera.set_window_size(m_width, m_height);
	m_camera.set_look_at(
		glm::vec3(2.0f, 2.0f, 2.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 0.0f, 1.0f)
	);
}

void BaseApplication::init_vulkan()
{
	auto res = volkInitialize();
	if (res != VK_SUCCESS) 
		throw std::runtime_error("could not initialize volk");

	create_instance();
	volkLoadInstance(m_instance);
	// check if compiler is initialized and valid
	if (!m_shader_compiler.IsValid()) 
		throw std::runtime_error("could not initialize shaderc compiler");

	if (m_enable_validation_layers) {
		setup_debug_callback();
	}

	create_surface();

	pick_gpu();
	create_logical_device();

	create_allocator();

	create_command_pools();
	create_sync_objects();

	create_swapchain();
	create_image_views();
	create_depth_resources();
	create_rt_image();

	create_render_pass();
	create_descriptor_set_layout();
	create_graphics_pipeline();
	create_framebuffers();

	// rt
	create_raytracing_pipeline_layout();
	create_raytracing_pipeline();

	load_model();

	create_vertex_buffer();
	create_index_buffer();
	create_uniform_buffers();

	create_spheres();
	create_sphere_buffer();

	create_bottom_acceleration_structure();
	create_bottom_acceleration_structure_spheres();
	create_top_acceleration_structure();

	create_descriptor_pool();
	create_descriptor_sets();
	create_rt_descriptor_sets();
	create_shader_binding_table();

	create_command_buffers();
	create_rt_command_buffers();
}

void BaseApplication::recreate_swapchain()
{
	int width = 0, height = 0;
	// we need to wait for the app to be in the foreground after minimizing
	while (width == 0 || height == 0) {
		glfwGetFramebufferSize(m_window, &width, &height);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(m_device);

	cleanup_swapchain();

	create_swapchain();
	create_image_views();
	create_depth_resources();
	create_rt_image();

	create_render_pass();
	create_descriptor_set_layout();
	create_graphics_pipeline();
	create_framebuffers();
	create_uniform_buffers();

	create_descriptor_pool();
	create_descriptor_sets();
	create_rt_descriptor_sets();

	create_command_buffers();
	create_rt_command_buffers();
}

void BaseApplication::main_loop()
{
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
		draw_frame();
	}
	vkDeviceWaitIdle(m_device);
}

void BaseApplication::cleanup_swapchain()
{
	if (!m_device) return;

	for (auto b : m_uni_buffers) {
		vmaDestroyBuffer(m_allocator, b.buffer, b.alloc);
	}
	
	// no need to free desc sets because we destroy the pool
	vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
	
	if (m_graphics_cmd_pool) {
		if (m_rt_cmd_buffers.size()) {
			vkFreeCommandBuffers(m_device, m_graphics_cmd_pool,
				static_cast<uint32_t>(m_rt_cmd_buffers.size()), m_rt_cmd_buffers.data());
		}
		if (m_cmd_buffers.size()) {
			vkFreeCommandBuffers(m_device, m_graphics_cmd_pool,
				static_cast<uint32_t>(m_cmd_buffers.size()), m_cmd_buffers.data());
		}
	}

	for (auto fb : m_swapchain_fbs) {
		vkDestroyFramebuffer(m_device, fb, nullptr);
	}
	
	vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
	vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
	vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
	vkDestroyRenderPass(m_device, m_render_pass, nullptr);
	
	vkDestroyImageView(m_device, m_depth_img_view, nullptr);
	vmaDestroyImage(m_allocator, m_depth_img.image, m_depth_img.alloc);
	
	// cleanup raytracing stuff
	vkDestroyImageView(m_device, m_rt_img_view, nullptr);
	vmaDestroyImage(m_allocator, m_rt_img.image, m_rt_img.alloc);
	
	for (auto img_view : m_swapchain_img_views) {
		vkDestroyImageView(m_device, img_view, nullptr);
	}
	vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
}

void BaseApplication::cleanup()
{
	cleanup_swapchain();

	if (m_device) {
		// cleanup raytracing stuff
		vkDestroyDescriptorSetLayout(m_device, m_rt_descriptor_set_layout, nullptr);
		vkDestroyPipelineLayout(m_device, m_rt_pipeline_layout, nullptr);
		vkDestroyPipeline(m_device, m_rt_pipeline, nullptr);
	}

	if (m_device && m_allocator) {
		// cleanup buffers and acceleration structures
		vmaDestroyBuffer(m_allocator, m_index_buffer.buffer, m_index_buffer.alloc);
		vmaDestroyBuffer(m_allocator, m_vertex_buffer.buffer, m_vertex_buffer.alloc);
		vmaDestroyBuffer(m_allocator, m_sphere_buffer.buffer, m_sphere_buffer.alloc);
		vmaDestroyBuffer(m_allocator, m_rt_sbt.buffer, m_rt_sbt.alloc);
		m_top_as.destroy(m_device, m_allocator);
		m_bottom_as.destroy(m_device, m_allocator);
		m_bottom_as_spheres.destroy(m_device, m_allocator);
		vmaDestroyAllocator(m_allocator);
	}

	if (m_device) {
		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroyFence(m_device, m_fen_flight[i], nullptr);
			vkDestroySemaphore(m_device, m_sem_img_available[i], nullptr);
			vkDestroySemaphore(m_device, m_sem_render_finished[i], nullptr);
		}
		vkDestroyCommandPool(m_device, m_transfer_cmd_pool, nullptr);
		vkDestroyCommandPool(m_device, m_graphics_cmd_pool, nullptr);
	}

	vkDestroyDevice(m_device, nullptr);

	if (m_instance) 
		vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
	
	destroy_debug_callback();
	
	vkDestroyInstance(m_instance, nullptr);

	if (m_window) 
		glfwDestroyWindow(m_window);
	
	glfwTerminate();
}

void BaseApplication::create_instance()
{
	if (m_enable_validation_layers && !check_validation_layer_support()) {
		throw std::runtime_error("requested validation layers not available");
	}

	VkApplicationInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	ai.pApplicationName = "Hello triangle";
	ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	ai.pEngineName = "-";
	ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	ai.apiVersion = VK_API_VERSION_1_2;

	auto required_exts = get_required_instance_extensions();

	VkInstanceCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	ci.pApplicationInfo = &ai;
	ci.enabledExtensionCount = static_cast<uint32_t>(required_exts.size());
	ci.ppEnabledExtensionNames = required_exts.data();
	if (m_enable_validation_layers) {
		ci.enabledLayerCount = static_cast<uint32_t>(m_validation_layers.size());
		ci.ppEnabledLayerNames = m_validation_layers.data();
	} else {
		ci.enabledLayerCount = 0;
	}
	
	auto res = vkCreateInstance(&ci, nullptr, &m_instance);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create instance");
	}
}

bool BaseApplication::check_validation_layer_support() const
{
	uint32_t layer_count;
	vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
	if (!layer_count) return false;
	std::vector<VkLayerProperties> avail_layers(layer_count);
	vkEnumerateInstanceLayerProperties(&layer_count, avail_layers.data());

	for (const char *layer_name : m_validation_layers) {
		bool found = false;
		for (const auto &layer_props : avail_layers) {
			if (std::strcmp(layer_name, layer_props.layerName) == 0) {
				found = true;
				break;
			}
		}
		if (!found) { return false; }
	}
	return true;
}

std::vector<const char*> BaseApplication::get_required_instance_extensions() const
{
	uint32_t glfw_ext_count = 0;
	const char **glfw_exts;
	glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

	std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);

	if (m_enable_validation_layers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}

void BaseApplication::setup_debug_callback()
{
	VkDebugUtilsMessengerCreateInfoEXT ci = {};
	ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	ci.messageSeverity =
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
	ci.messageType =
		VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
	ci.pfnUserCallback = debug_callback;
	ci.pUserData = nullptr;

	auto res = vkCreateDebugUtilsMessengerEXT(m_instance, &ci, nullptr, &m_debug_callback);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to setup debug callback");
	}
}

void BaseApplication::destroy_debug_callback()
{
	if (!m_debug_callback) return;
	vkDestroyDebugUtilsMessengerEXT(m_instance, m_debug_callback, nullptr);
}

void BaseApplication::pick_gpu()
{
	uint32_t gpu_count = 0;
	vkEnumeratePhysicalDevices(m_instance, &gpu_count, nullptr);
	if (gpu_count == 0) {
		throw std::runtime_error("failed to find at least one GPU with Vulkan support");
	}
	std::vector<VkPhysicalDevice> gpus(gpu_count);
	vkEnumeratePhysicalDevices(m_instance, &gpu_count, gpus.data());
	
	for (const auto &gpu : gpus) {
		if (is_gpu_suitable(gpu)) {
			m_gpu = gpu; break;
		}
	}
	if (m_gpu == VK_NULL_HANDLE) {
		throw std::runtime_error("failed to find at least one suitable GPU");
	}
}

bool BaseApplication::check_device_extension_support(VkPhysicalDevice gpu) const
{
	uint32_t ext_count = 0;
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ext_count, nullptr);
	std::vector<VkExtensionProperties> available_exts(ext_count);
	vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ext_count, available_exts.data());

	std::set<std::string> required_exts(m_device_extensions.begin(), m_device_extensions.end());
	// if the required extension is supported it will be checked off of the set
	// so if in the end the set is empty then all requirements are fullfilled
	for (const auto &ext : available_exts) {
		required_exts.erase(ext.extensionName);
	}
	
	return required_exts.empty();
}

bool BaseApplication::is_gpu_suitable(VkPhysicalDevice gpu) const
{
	VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features = {};
	as_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	VkPhysicalDeviceRayQueryFeaturesKHR rq_features = {};
	rq_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
	rq_features.pNext = &as_features;
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtp_features = {};
	rtp_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	rtp_features.pNext = &rq_features;
	VkPhysicalDeviceVulkan12Features v12_features = {};
	v12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	v12_features.pNext = &rtp_features;
	VkPhysicalDeviceSynchronization2FeaturesKHR sh2 = {};
	sh2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
	sh2.pNext = &v12_features;
	VkPhysicalDeviceFeatures2 features;
	features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	features.pNext = &sh2;
	vkGetPhysicalDeviceFeatures2(gpu, &features);
	
	auto indices = find_queue_families(gpu);

	bool extensions_supported = check_device_extension_support(gpu);
	bool swapchain_adequate = false;
	if (extensions_supported) {
		auto chain_details = query_swapchain_support(gpu);
		swapchain_adequate = !chain_details.formats.empty() && 
			!chain_details.present_modes.empty();
	}

	bool supported_features =
		features.features.vertexPipelineStoresAndAtomics &&
		features.features.samplerAnisotropy &&
		rtp_features.rayTracingPipeline &&
		rq_features.rayQuery &&
		as_features.accelerationStructure &&
		v12_features.bufferDeviceAddress &&
		sh2.synchronization2;
	
	return indices.is_complete() && extensions_supported && supported_features && swapchain_adequate;
}

QueueFamilyIndices BaseApplication::find_queue_families(VkPhysicalDevice gpu) const
{
	QueueFamilyIndices indices;

	uint32_t family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &family_count, families.data());
	
	int i = 0;
	for (const auto &family : families) {
		VkBool32 present_support = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, m_surface, &present_support);
		if (family.queueCount > 0 && present_support) {
			indices.present_family = i;
		}
		if (family.queueCount > 0 && family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.graphics_family = i;
		}
		if (family.queueCount > 0 && (family.queueFlags & VK_QUEUE_TRANSFER_BIT)
				&& !(family.queueFlags & VK_QUEUE_COMPUTE_BIT)
				&& !(family.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
			indices.transfer_family = i;
		}
		if (indices.is_complete()) {
			break;
		}
		i++;
	}

	return indices;
}

void BaseApplication::create_logical_device()
{
	auto family_indices = find_queue_families(m_gpu);

	std::set<uint32_t> unique_queue_families = {
		family_indices.graphics_family.value(),
		family_indices.present_family.value(),
		family_indices.transfer_family.value()
	};
	
	std::vector<VkDeviceQueueCreateInfo> qcis;
	float queue_priority = 1.0f;
	for (uint32_t fidx : unique_queue_families) {
		VkDeviceQueueCreateInfo qci = {};
		qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		qci.queueFamilyIndex = fidx;
		qci.queueCount = 1;
		qci.pQueuePriorities = &queue_priority;
		qcis.push_back(qci);
	}

	VkPhysicalDeviceAccelerationStructureFeaturesKHR dasf = {};
	dasf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	dasf.accelerationStructure = VK_TRUE;

	VkPhysicalDeviceRayQueryFeaturesKHR drqf = {};
	drqf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
	drqf.rayQuery = VK_TRUE;
	drqf.pNext = &dasf;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR drtf = {};
	drtf.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	drtf.rayTracingPipeline = VK_TRUE;
	drtf.pNext = &drqf;

	VkPhysicalDeviceVulkan12Features v12f = {};
	v12f.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	v12f.bufferDeviceAddress = VK_TRUE;
	v12f.pNext = &drtf;

	VkPhysicalDeviceSynchronization2FeaturesKHR sh2 = {};
	sh2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
	sh2.synchronization2 = VK_TRUE;
	sh2.pNext = &v12f;

	VkPhysicalDeviceFeatures2 df = {};
	df.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
	df.features.samplerAnisotropy = VK_TRUE;
	df.features.vertexPipelineStoresAndAtomics = VK_TRUE;
	df.pNext = &sh2;

	VkDeviceCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	ci.pNext = &df;
	ci.pQueueCreateInfos = qcis.data();
	ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
	ci.pEnabledFeatures = nullptr;

	ci.enabledExtensionCount = static_cast<uint32_t>(m_device_extensions.size());
	ci.ppEnabledExtensionNames = m_device_extensions.data();
	if (m_enable_validation_layers) {
		ci.enabledLayerCount = static_cast<uint32_t>(m_validation_layers.size());
		ci.ppEnabledLayerNames = m_validation_layers.data();
	} else {
		ci.enabledLayerCount = 0;
	}
	auto res = vkCreateDevice(m_gpu, &ci, nullptr, &m_device);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create logical device");
	}

	vkGetDeviceQueue(m_device, family_indices.graphics_family.value(), 0, &m_graphics_queue);
	vkGetDeviceQueue(m_device, family_indices.present_family.value(), 0, &m_present_queue);
	vkGetDeviceQueue(m_device, family_indices.transfer_family.value(), 0, &m_transfer_queue);
}

void BaseApplication::create_allocator()
{
	// we use volk so we need to provide their functions to vma 
	VmaVulkanFunctions funcs = {};
	funcs.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
	funcs.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
	funcs.vkAllocateMemory = vkAllocateMemory;
	funcs.vkFreeMemory = vkFreeMemory;
	funcs.vkMapMemory = vkMapMemory;
	funcs.vkUnmapMemory = vkUnmapMemory;
	funcs.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
	funcs.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
	funcs.vkBindBufferMemory = vkBindBufferMemory;
	funcs.vkBindImageMemory = vkBindImageMemory;
	funcs.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
	funcs.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
	funcs.vkCreateBuffer = vkCreateBuffer;
	funcs.vkDestroyBuffer = vkDestroyBuffer;
	funcs.vkCreateImage = vkCreateImage;
	funcs.vkDestroyImage = vkDestroyImage;
	funcs.vkCmdCopyBuffer = vkCmdCopyBuffer;
	funcs.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
	funcs.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
	funcs.vkBindBufferMemory2KHR = vkBindBufferMemory2;
	funcs.vkBindImageMemory2KHR = vkBindImageMemory2;
	funcs.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;

	VmaAllocatorCreateInfo ai = {};
	ai.vulkanApiVersion = VK_API_VERSION_1_2;
	ai.physicalDevice = m_gpu;
	ai.device = m_device;
	ai.instance = m_instance;
	ai.pVulkanFunctions = &funcs;
	ai.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	auto res = vmaCreateAllocator(&ai, &m_allocator);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("could not create vma allocator");
	}
}

void BaseApplication::create_surface()
{
	auto res = glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create window surface");
	}
}

SwapchainSupportDetails BaseApplication::query_swapchain_support(VkPhysicalDevice gpu) const
{
	SwapchainSupportDetails details;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, m_surface, &details.capabilities);
	
	uint32_t format_count;
	vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_surface, &format_count, nullptr);
	if (format_count != 0) {
		details.formats.resize(format_count);
		vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, m_surface, 
			&format_count, details.formats.data());
	}

	uint32_t present_mode_count;
	vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_surface, &present_mode_count, nullptr);
	if (present_mode_count != 0) {
		details.present_modes.resize(present_mode_count);
		vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, m_surface,
											 &present_mode_count, details.present_modes.data());
	}

	return details;
}

VkSurfaceFormatKHR BaseApplication::choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) const
{
	if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
		// best case scenario
		return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
	}

	for (const auto &f : formats) {
		if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
			return f;
		}
	}

	return formats[0];
}

VkPresentModeKHR BaseApplication::choose_swap_present_mode(const std::vector<VkPresentModeKHR> modes) const
{
	// vulkan guarantees that fifo is available
	VkPresentModeKHR best_mode = VK_PRESENT_MODE_FIFO_KHR;

	for (const auto &m : modes) {
		// check if we can use triple buffering
		if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
			return m;
		} else if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) {
			// some drivers do not properly support fifo so if immediate is available choose this
			best_mode = m;
		}
	}
	return best_mode;
}

VkExtent2D BaseApplication::choose_swap_extent(const VkSurfaceCapabilitiesKHR & capabilities) const
{
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	} else {
		int width, height;
		glfwGetFramebufferSize(m_window, &width, &height);
		VkExtent2D actual = { uint32_t(width), uint32_t(height) };
		
		uint32_t min_width = capabilities.minImageExtent.width;
		uint32_t max_width = capabilities.maxImageExtent.width;
		uint32_t min_height = capabilities.minImageExtent.height;
		uint32_t max_height = capabilities.maxImageExtent.height;
		actual.width = std::max(min_width, std::min(max_width, actual.width));
		actual.height = std::max(min_height, std::min(max_height, actual.height));
		return actual;
	}
}

void BaseApplication::create_swapchain()
{
	auto details = query_swapchain_support(m_gpu);
	auto format = choose_swap_surface_format(details.formats);
	auto present_mode = choose_swap_present_mode(details.present_modes);
	auto extent = choose_swap_extent(details.capabilities);
	// triple buf
	uint32_t img_count = details.capabilities.minImageCount + 1;
	// a value of 0 for maxImageCount means unlimited
	if (details.capabilities.maxImageCount > 0 && img_count > details.capabilities.maxImageCount) {
		img_count = details.capabilities.maxImageCount;
	}

	auto family_indices = find_queue_families(m_gpu);
	uint32_t queue_family_indices[] = {
		family_indices.graphics_family.value(),
		family_indices.present_family.value(),
	};

	VkSwapchainCreateInfoKHR ci = {};
	ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	ci.surface = m_surface;
	ci.minImageCount = img_count;
	ci.imageFormat = format.format;
	ci.imageColorSpace = format.colorSpace;
	ci.imageExtent = extent;
	ci.imageArrayLayers = 1; // this is for stereo
	ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	if (family_indices.graphics_family != family_indices.present_family) {
		ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		ci.queueFamilyIndexCount = 2;
		ci.pQueueFamilyIndices = queue_family_indices;
	} else {
		ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		ci.queueFamilyIndexCount = 0; // optional
		ci.pQueueFamilyIndices = nullptr; // optional
	}
	ci.preTransform = details.capabilities.currentTransform;
	ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	ci.presentMode = present_mode;
	ci.clipped = VK_TRUE;
	ci.oldSwapchain = VK_NULL_HANDLE;

	auto res = vkCreateSwapchainKHR(m_device, &ci, nullptr, &m_swapchain);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create swapchain");
	}

	// store images created and owned by the swapchain
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &img_count, nullptr);
	m_swapchain_images.resize(img_count);
	vkGetSwapchainImagesKHR(m_device, m_swapchain, &img_count, m_swapchain_images.data());

	// store format and extent
	m_swapchain_extent = extent;
	m_swapchain_img_format = format.format;

	// also change current width and height
	m_width = extent.width;
	m_height = extent.height;
}

void BaseApplication::create_image_views()
{
	m_swapchain_img_views.resize(m_swapchain_images.size());
	for (size_t i = 0; i < m_swapchain_images.size(); ++i) {
		m_swapchain_img_views[i] = vk_helpers::create_image_view_2d(m_device, 
										m_swapchain_images[i], m_swapchain_img_format, VK_IMAGE_ASPECT_COLOR_BIT);
	}
}

void BaseApplication::create_render_pass()
{
	VkAttachmentDescription2 color_attachment = {};
	color_attachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
	color_attachment.format = m_swapchain_img_format;
	color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	// clear the values to a constant at the start
	color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
	// rendered contents will be stored in memory and can be read later
	color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentDescription2 depth_attachment = {};
	depth_attachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
	depth_attachment.format = find_supported_depth_format();
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference2 color_attachment_ref = {};
	color_attachment_ref.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
	color_attachment_ref.attachment = 0; // index to above descriptions
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color_attachment_ref.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

	VkAttachmentReference2 depth_attachment_ref = {};
	depth_attachment_ref.sType = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depth_attachment_ref.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	
	VkSubpassDescription2 subpass = {};
	subpass.sType = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	// the index in this array is referenced in the frag shader
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	VkMemoryBarrier2KHR mem_bar = {};
	mem_bar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR;
	// srcStageMask needs to be a part of pWaitDstStageMask in the WSI semaphore.
	mem_bar.srcStageMask = 
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR |
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR;
	mem_bar.dstStageMask = 
		VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR |
		VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT_KHR;
	mem_bar.srcAccessMask = 0;
	mem_bar.dstAccessMask = 
		VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT_KHR |
		VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT_KHR;

	VkSubpassDependency2 dependency = {};
	dependency.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
	dependency.pNext = &mem_bar;
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0; // our subpass
	
	std::array<VkAttachmentDescription2, 2> attachments = {
		color_attachment, depth_attachment
	};
	VkRenderPassCreateInfo2 rpci = {};
	rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
	rpci.attachmentCount = uint32_t(attachments.size());
	rpci.pAttachments = attachments.data();
	rpci.subpassCount = 1;
	rpci.pSubpasses = &subpass;
	rpci.dependencyCount = 1;
	rpci.pDependencies = &dependency;

	auto res = vkCreateRenderPass2(m_device, &rpci, nullptr, &m_render_pass);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass");
	}
}

void BaseApplication::create_descriptor_set_layout()
{
	VkDescriptorSetLayoutBinding lb = {};
	lb.binding = 0;
	lb.descriptorCount = 1;
	lb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	lb.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	lb.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding lb_1 = {};
    lb_1.binding = 1;
    lb_1.descriptorCount = 1;
    lb_1.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    lb_1.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    lb_1.pImmutableSamplers = nullptr;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
		lb, lb_1
	};

	VkDescriptorSetLayoutCreateInfo li = {};
	li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	li.bindingCount = uint32_t(bindings.size());
	li.pBindings = bindings.data();
	auto res = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_descriptor_set_layout);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create descriptor set layout");
}

void BaseApplication::create_graphics_pipeline()
{
	// 1. Shader modules
	auto vert_code = read_file(SHADER_DIR "simple.vert");
	auto frag_code = read_file(SHADER_DIR "simple.frag");
	auto vert_module = create_shader_module("simple.vert", shaderc_vertex_shader, vert_code);
	auto frag_module = create_shader_module("simple.frag", shaderc_fragment_shader, frag_code);

	VkPipelineShaderStageCreateInfo vsci = {};
	vsci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	vsci.stage = VK_SHADER_STAGE_VERTEX_BIT;
	vsci.module = vert_module;
	vsci.pName = "main";

	VkPipelineShaderStageCreateInfo fsci = {};
	fsci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	fsci.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	fsci.module = frag_module;
	fsci.pName = "main";

	VkPipelineShaderStageCreateInfo shader_stages[] = {
		vsci, fsci
	};

	// 2. Vertex Input 
	auto binding_desc = Vertex::get_binding_description();
	auto attrib_desc = Vertex::get_attribute_descriptions();

	VkPipelineVertexInputStateCreateInfo vici = {};
	vici.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vici.vertexBindingDescriptionCount = 1;
	vici.pVertexBindingDescriptions = &binding_desc;
	vici.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrib_desc.size());
	vici.pVertexAttributeDescriptions = attrib_desc.data();

	// 3. Input Assembly
	VkPipelineInputAssemblyStateCreateInfo iaci = {};
	iaci.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	iaci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	iaci.primitiveRestartEnable = VK_FALSE;

	// 4. Viewports and Scissors
	VkViewport viewport = {};
	viewport.x = 0.0f;
	viewport.y = 0.0f;
	viewport.width = (float)m_swapchain_extent.width;
	viewport.height = (float)m_swapchain_extent.height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	VkRect2D scissor = {};
	scissor.offset = { 0, 0 };
	scissor.extent = m_swapchain_extent;

	VkPipelineViewportStateCreateInfo vci = {};
	vci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	vci.viewportCount = 1;
	vci.pViewports = &viewport;
	vci.scissorCount = 1;
	vci.pScissors = &scissor;

	// 5. Rasterizer
	VkPipelineRasterizationStateCreateInfo rci = {};
	rci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rci.depthClampEnable = VK_FALSE;
	rci.rasterizerDiscardEnable = VK_FALSE; // other requires GPU feature
	rci.polygonMode = VK_POLYGON_MODE_FILL; // other requires GPU feature
	rci.lineWidth = 1.0f; // other requires GPU feature
	rci.cullMode = VK_CULL_MODE_BACK_BIT;
	rci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rci.depthBiasEnable = VK_FALSE; // polygon offset
	rci.depthBiasConstantFactor = 0.0f; // optional
	rci.depthBiasClamp = 0.0f; // optional;
	rci.depthBiasSlopeFactor = 0.0f; // optional

	// 6. Multisampling
	VkPipelineMultisampleStateCreateInfo msci = {};
	msci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	msci.sampleShadingEnable = VK_FALSE;
	msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	
	// 7. Depth Stencil
	VkPipelineDepthStencilStateCreateInfo ds = {};
	ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	ds.depthTestEnable = VK_TRUE;
	ds.depthWriteEnable = VK_TRUE;
	ds.depthCompareOp = VK_COMPARE_OP_LESS;
	ds.depthBoundsTestEnable = VK_FALSE;
	ds.minDepthBounds = 0.0f; // optional
	ds.maxDepthBounds = 1.0f; // optional
	ds.stencilTestEnable = VK_FALSE;
	ds.front = {}; // optional
	ds.back = {}; // optional

	// 8. Color blending 
	
	// per fb attachment info 
	VkPipelineColorBlendAttachmentState cba = {};
	cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	cba.blendEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo cbci = {};
	cbci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	cbci.logicOpEnable = VK_FALSE;
	cbci.logicOp = VK_LOGIC_OP_COPY;
	cbci.attachmentCount = 1;
	cbci.pAttachments = &cba;
	
#if 0
	// 9. Dynamic State
	VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_LINE_WIDTH
	};
	VkPipelineDynamicStateCreateInfo dci = {};
	dci.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dci.dynamicStateCount = 2;
	dci.pDynamicStates = dynamic_states;
#endif

	// 10. Pipeline Layout
	VkPipelineLayoutCreateInfo plci = {};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &m_descriptor_set_layout;
	plci.pushConstantRangeCount = 0;
	plci.pPushConstantRanges = nullptr;

	auto res = vkCreatePipelineLayout(m_device, &plci, nullptr, &m_pipeline_layout);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout");
	}

	VkGraphicsPipelineCreateInfo pci = {};
	pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pci.stageCount = 2;
	pci.pStages = shader_stages;
	pci.pVertexInputState = &vici;
	pci.pInputAssemblyState = &iaci;
	pci.pViewportState = &vci;
	pci.pRasterizationState = &rci;
	pci.pMultisampleState = &msci;
	pci.pDepthStencilState = &ds;
	pci.pColorBlendState = &cbci;
	pci.pDynamicState = nullptr;
	pci.layout = m_pipeline_layout;
	pci.renderPass = m_render_pass;
	pci.subpass = 0;
	pci.basePipelineHandle = VK_NULL_HANDLE; // optional
	pci.basePipelineIndex = -1; // optional

	res = vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pci, nullptr, &m_graphics_pipeline);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create graphics pipeline");
	}

	// at last destroy shader modules
	vkDestroyShaderModule(m_device, vert_module, nullptr);
	vkDestroyShaderModule(m_device, frag_module, nullptr);
}

class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface
{
public:
	virtual shaderc_include_result* GetInclude(const char* requested_source,
		shaderc_include_type type,
		const char* requesting_source,
		size_t include_depth) override
	{
		assert(type == shaderc_include_type_relative);
		std::string filename = SHADER_DIR + std::string(requested_source);
		auto data = read_file(filename.c_str());
		std::string source { data.begin(), data.end() };
		auto [it, inserted] = m_includes.insert({ filename, source });
		return new shaderc_include_result{
			it->first.c_str(),
			it->first.size(),
			it->second.c_str(),
			it->second.size(),
			nullptr
		};
	}

	virtual void ReleaseInclude(shaderc_include_result* data) override
	{
		delete data;
		// the map will free the real data in the destructor
	}

private:
	std::unordered_map<std::string, std::string> m_includes;
};

VkShaderModule BaseApplication::create_shader_module(const std::string &file_name, 
	shaderc_shader_kind shader_kind, const std::vector<char>& code) const
{
	shaderc::CompileOptions opts;
	opts.SetGenerateDebugInfo();
	opts.SetOptimizationLevel(shaderc_optimization_level_zero);
	opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
	opts.SetIncluder(std::make_unique<ShaderIncluder>());
	
	std::string source{ code.begin(), code.end() };
	auto result = m_shader_compiler.CompileGlslToSpv(source, shader_kind, file_name.c_str(), opts);
	
	if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
		fprintf(stdout, "SHADERC COMPILE ERROR\n%s", result.GetErrorMessage().c_str());
		throw std::runtime_error("failed to compile shader");
	}
	
	VkShaderModuleCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = (result.cend() - result.cbegin()) * sizeof(uint32_t);
	// cast bytes to uints
	ci.pCode = result.cbegin();
	VkShaderModule shader_module;
	auto res = vkCreateShaderModule(m_device, &ci, nullptr, &shader_module);

	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create shader module");
	}
	return shader_module;
}

void BaseApplication::create_framebuffers()
{
	m_swapchain_fbs.resize(m_swapchain_img_views.size());
	for (size_t i = 0; i < m_swapchain_img_views.size(); ++i) {
		std::array<VkImageView, 2> attachments = {
			m_swapchain_img_views[i],
			m_depth_img_view
		};
		VkFramebufferCreateInfo fbci = {};
		fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		fbci.renderPass = m_render_pass;
		fbci.attachmentCount = uint32_t(attachments.size());
		fbci.pAttachments = attachments.data();
		fbci.width = m_swapchain_extent.width;
		fbci.height = m_swapchain_extent.height;
		fbci.layers = 1;

		auto res = vkCreateFramebuffer(m_device, &fbci, nullptr, &m_swapchain_fbs[i]);
		if (res != VK_SUCCESS) {
			throw std::runtime_error("failed to create framebuffer");
		}
	}
}

uint32_t BaseApplication::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const
{
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(m_gpu, &mem_props);
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
		if (type_filter & (1 << i) && 
				(mem_props.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type");
	return 0;
}

void BaseApplication::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
									VkMemoryPropertyFlags props, 
									VmaBufferAllocation &buffer)
{
	auto indices = find_queue_families(m_gpu);
	uint32_t qidx[] = {
		indices.graphics_family.value(),
		indices.transfer_family.value(),
	};

	VkBufferCreateInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bi.size = size;
	bi.usage = usage;
	bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
	bi.queueFamilyIndexCount = 2;
	bi.pQueueFamilyIndices = qidx;
	
	VmaAllocationCreateInfo ai = {};
	if (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
		ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	} else {
		ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	}

	auto res = vmaCreateBuffer(m_allocator, &bi, &ai, &buffer.buffer, &buffer.alloc, nullptr);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create & allocate buffer");
}

void BaseApplication::copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
	auto cmd_buf = begin_single_time_commands(m_transfer_queue, m_transfer_cmd_pool);

	VkBufferCopy cpy = {};
	cpy.srcOffset = 0;
	cpy.dstOffset = 0;
	cpy.size = size;
	
	vkCmdCopyBuffer(cmd_buf, src, dst, 1, &cpy);

	end_single_time_commands(m_transfer_queue, m_transfer_cmd_pool, cmd_buf);
}

void BaseApplication::create_image(uint32_t width, uint32_t height, VkFormat format, 
								   VkImageTiling tiling, VkImageUsageFlags usage, 
								   VkMemoryPropertyFlags props, VmaImageAllocation &img)
{
	VkImageCreateInfo ii = {};
	ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	ii.imageType = VK_IMAGE_TYPE_2D;
	ii.extent = { width, height, 1 };
	ii.mipLevels = 1;
	ii.arrayLayers = 1;
	ii.format = format;
	ii.tiling = tiling;
	ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	ii.usage = usage;
	ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	ii.samples = VK_SAMPLE_COUNT_1_BIT;
	ii.flags = 0;

	VmaAllocationCreateInfo ai = {};
	if (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
		ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	} else {
		ai.usage = VMA_MEMORY_USAGE_CPU_ONLY;
	}

	auto res = vmaCreateImage(m_allocator, &ii, &ai, &img.image, &img.alloc, nullptr);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create image");
}

VkCommandBuffer BaseApplication::begin_single_time_commands(VkQueue queue, VkCommandPool cmd_pool)
{
	VkCommandBufferAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	ai.commandPool = cmd_pool;
	ai.commandBufferCount = 1;

	VkCommandBuffer cmd_buf;
	auto res = vkAllocateCommandBuffers(m_device, &ai, &cmd_buf);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate command buffer");
	VkCommandBufferBeginInfo bi = {};
	bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	res = vkBeginCommandBuffer(cmd_buf, &bi);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to begin command buffer recording");
	return cmd_buf;
}

void BaseApplication::end_single_time_commands(VkQueue queue, VkCommandPool cmd_pool, VkCommandBuffer cmd_buffer)
{
	auto res = vkEndCommandBuffer(cmd_buffer);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to end command buffer recording");

	VkCommandBufferSubmitInfoKHR cmd_submit = {};
	cmd_submit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR;
	cmd_submit.commandBuffer = cmd_buffer;
	cmd_submit.deviceMask = 0;

	VkSubmitInfo2KHR submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR;
	submit_info.commandBufferInfoCount = 1;
	submit_info.pCommandBufferInfos = &cmd_submit;

	res = vkQueueSubmit2KHR(queue, 1, &submit_info, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to submit to queue");
	res  = vkQueueWaitIdle(queue);
	if (res == VK_ERROR_DEVICE_LOST) {
		printf("DEVICE LOST\n");
	}

	vkFreeCommandBuffers(m_device, cmd_pool, 1, &cmd_buffer);
}

#if 0
void BaseApplication::transition_image_layout(VkImage img, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout)
{
	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = old_layout;
	barrier.newLayout = new_layout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = img;
	if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
		if (vk_helpers::format_has_stencil_component(format)) {
			barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
		}
	} else {
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	}
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags src_stage;
	VkPipelineStageFlags dst_stage;
	if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
	}  else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	} else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		dst_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT; // earliest stage used
	} else {
		throw std::runtime_error("unsupported texture layout transition");
	}

	vkCmdPipelineBarrier(cmd_buf, src_stage, dst_stage,
						 0, 0, nullptr, 0, nullptr, 1, &barrier);

	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}
#endif

void BaseApplication::copy_buffer_to_image(VkBuffer buffer, VkImage img, uint32_t width, uint32_t height)
{
	VkCommandBuffer cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);
	
	VkBufferImageCopy rg = {};
	rg.bufferOffset = 0;
	rg.bufferRowLength = 0; // tightly packed
	rg.bufferImageHeight = 0; // tightly packed
	rg.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	rg.imageSubresource.mipLevel = 0;
	rg.imageSubresource.baseArrayLayer = 0;
	rg.imageSubresource.layerCount = 1;
	rg.imageOffset = { 0, 0, 0 };
	rg.imageExtent = { width, height, 1 };
	vkCmdCopyBufferToImage(cmd_buf, buffer, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);

	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::load_model()
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> parts;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &parts, &materials, &warn, &err, "resources/bmw.obj", "resources")) {
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
				attrib.vertices[3 * index.vertex_index + 0]+400,
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]+200,
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

        uint32_t vertex_offset = m_model_vertices.size();
        uint32_t index_offset = m_model_indices.size();
        m_model_vertices.insert(m_model_vertices.end(), part_vertices.begin(), part_vertices.end());
        m_model_indices.insert(m_model_indices.end(), part_indices.begin(), part_indices.end());
        
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
        m_model_parts.push_back(part_info);
        printf("Add part %s {v0 %u, vc %u, i0 %u, ic %u}\t material [albedo {%.2f, %.2f, %.2f, %.2f}, metallic %.2f, roughness %.2f\n",
			part.name.c_str(),
			part_info.vertex_offset, part_info.vertex_count, part_info.index_offset, part_info.index_count,
			part_info.pbr_material.albedo.r, part_info.pbr_material.albedo.g, 
			part_info.pbr_material.albedo.b, part_info.pbr_material.albedo.a,
			part_info.pbr_material.metallic, part_info.pbr_material.roughness);

	}
    fprintf(stdout, "Loaded model part: num vertices %" PRIu64 ", num indices %" PRIu64 "\n",
        m_model_vertices.size(),
        m_model_indices.size());

	auto [vxmin, vxmax] = std::minmax_element(m_model_vertices.begin(), m_model_vertices.end(), Vertex::compare_position<0>);
	auto [vymin, vymax] = std::minmax_element(m_model_vertices.begin(), m_model_vertices.end(), Vertex::compare_position<1>);
	auto [vzmin, vzmax] = std::minmax_element(m_model_vertices.begin(), m_model_vertices.end(), Vertex::compare_position<2>);
	glm::vec3 min_coord(vxmin->pos.x, vymin->pos.y, vzmin->pos.z);
	glm::vec3 max_coord(vxmax->pos.x, vymax->pos.y, vzmax->pos.z);
	glm::vec3 diff_coord = max_coord - min_coord;
	float scale = std::min({ diff_coord.x, diff_coord.y, diff_coord.z });
	glm::mat4 model_scale = glm::scale(glm::mat4(1.0), glm::vec3(1.0f / scale));

	// translate model to its centroid
	glm::vec3 centroid = (min_coord + max_coord) * 0.5f;
	glm::mat4 model_translate = glm::translate(glm::mat4(1.0), -centroid);

	// x axis is the car's major axis, y is up
	glm::mat4 model_rotate = glm::rotate(glm::mat4(1.0f), glm::half_pi<float>(), glm::vec3(0.0f, 0.0f, 1.0f));
	model_rotate = glm::rotate(model_rotate, glm::half_pi<float>(), glm::vec3(1.0, 0.0, 0.0));

	// y is up in car coordinates
	float half_height = diff_coord.y * 0.5f;
	half_height /= scale;
	glm::mat4 translate_to_ground = glm::translate(glm::mat4(1.0), glm::vec3(0.0f, 0.0f, half_height));

	m_model_tranformation = translate_to_ground * model_scale * model_rotate * model_translate;
}

void BaseApplication::create_spheres()
{
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);

	auto rgen = [&]() {return dist(engine); };
	const float scale = 0.3f;
#if 1
	for (int a = -10; a < 10; ++a) {
		for (int b = -10; b < 10; ++b) {
			SpherePrimitive sphere = {};
			float radius = 0.1 * glm::clamp(rgen(), 0.2f, 1.0f);
			glm::vec3 center = glm::vec3(scale*a + scale *rgen(), scale*b + scale*rgen(), +radius);
			// compute aabb
			glm::vec3 aabb_min = center - glm::vec3(radius);
			glm::vec3 aabb_max = center + glm::vec3(radius);
			sphere.bbox.minX = aabb_min.x;
			sphere.bbox.minY = aabb_min.y;
			sphere.bbox.minZ = aabb_min.z;
			sphere.bbox.maxX = aabb_max.x;
			sphere.bbox.maxY = aabb_max.y;
			sphere.bbox.maxZ = aabb_max.z;
			float material_rand = rgen();
			if (material_rand > 0.90) {
				sphere.material = materials::MaterialType::EMISSIVE;
			} else if (material_rand > 0.4) {
				sphere.material = materials::MaterialType::METAL;
			} else {
				sphere.material = materials::MaterialType::LAMBERTIAN;
			}
			if (sphere.material == materials::MaterialType::EMISSIVE) {
				const float light_intensity = rgen() * 50;
				sphere.albedo = glm::vec4(light_intensity*rgen(), light_intensity*rgen(), light_intensity*rgen(), 1.0f);
			} else {
				sphere.albedo = glm::vec4(rgen(), rgen(), rgen(), 1.0f);
			}
			sphere.fuzz = rgen();
			m_sphere_primitives.push_back(sphere);
		}
	}
	// add a really big one
	{
		SpherePrimitive earth = {};
		glm::vec3 center = glm::vec3(0.0, 0.0, -3000-0.01);
		float radius = 3000;
		glm::vec3 aabb_min = center - glm::vec3(radius);
		glm::vec3 aabb_max = center + glm::vec3(radius);
		earth.bbox.minX = aabb_min.x;
		earth.bbox.minY = aabb_min.y;
		earth.bbox.minZ = aabb_min.z;
		earth.bbox.maxX = aabb_max.x;
		earth.bbox.maxY = aabb_max.y;
		earth.bbox.maxZ = aabb_max.z;
		earth.albedo = glm::vec4(0.2f, 0.4f, 0.6f, 1.0f);
		earth.material = materials::MaterialType::LAMBERTIAN;
		m_sphere_primitives.push_back(earth);
	}
	
#else
	SpherePrimitive sphere;
	sphere.aabb_min = { -1.0f, -1.0f, -1.0f };
	sphere.aabb_max = { -0.5f , -0.5f, -0.5f };
	sphere.albedo = glm::vec4(rgen(), rgen(), rgen(), 1.0f);
	m_sphere_primitives.push_back(sphere);
#endif
}

void BaseApplication::create_vertex_buffer()
{
	auto bufsize = sizeof(Vertex) * m_model_vertices.size();
	
	VmaBufferAllocation staging;
	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				  staging);

	void *data;
	auto res = vmaMapMemory(m_allocator, staging.alloc, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
	std::memcpy(data, m_model_vertices.data(), bufsize);
	vmaUnmapMemory(m_allocator, staging.alloc);

	create_buffer(bufsize, 
		VK_BUFFER_USAGE_TRANSFER_DST_BIT | 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | 
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | 
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertex_buffer);
	copy_buffer(staging.buffer, m_vertex_buffer.buffer, bufsize);

	vmaDestroyBuffer(m_allocator, staging.buffer, staging.alloc);
}

void BaseApplication::create_index_buffer()
{
	VkDeviceSize bufsize = sizeof(uint32_t) * m_model_indices.size();
	
	VmaBufferAllocation staging;
	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				  staging);

	void *data;
	auto res = vmaMapMemory(m_allocator, staging.alloc, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
	std::memcpy(data, m_model_indices.data(), bufsize);
	vmaUnmapMemory(m_allocator, staging.alloc);

	create_buffer(bufsize,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_index_buffer);
	copy_buffer(staging.buffer, m_index_buffer.buffer, bufsize);

	vmaDestroyBuffer(m_allocator, staging.buffer, staging.alloc);
}

void BaseApplication::create_uniform_buffers()
{
	VkDeviceSize bufsize = sizeof(SceneUniforms);
	m_uni_buffers.resize(m_swapchain_images.size());

	for (size_t i = 0; i < m_swapchain_images.size(); ++i) {
		create_buffer(bufsize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					  m_uni_buffers[i]);
	}
}

void BaseApplication::create_sphere_buffer()
{
	auto bufsize = sizeof(SpherePrimitive) * m_sphere_primitives.size();
	
	VmaBufferAllocation staging;
	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
		staging);
	
	void *data;
	auto res = vmaMapMemory(m_allocator, staging.alloc, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
	std::memcpy(data, m_sphere_primitives.data(), bufsize);
	vmaUnmapMemory(m_allocator, staging.alloc);

	create_buffer(bufsize,
		VK_BUFFER_USAGE_TRANSFER_DST_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_sphere_buffer);
	copy_buffer(staging.buffer, m_sphere_buffer.buffer, bufsize);

	vmaDestroyBuffer(m_allocator, staging.buffer, staging.alloc);
}

void BaseApplication::create_bottom_acceleration_structure()
{
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> max_primitive_counts;
    for (const ModelPart &part : m_model_parts) {
        VkAccelerationStructureGeometryKHR geom = {};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

        VkAccelerationStructureGeometryTrianglesDataKHR &geom_trias = geom.geometry.triangles;
        geom_trias.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom_trias.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom_trias.vertexStride = sizeof(Vertex);
        geom_trias.indexType = VK_INDEX_TYPE_UINT32;
        geom_trias.maxVertex = part.vertex_count - 1;
        // for now 
        geom_trias.vertexData.deviceAddress = 0;
        geom_trias.indexData.deviceAddress = 0;
        geom_trias.transformData.deviceAddress = 0;

        geometries.push_back(geom);
        max_primitive_counts.push_back(part.index_count/3);
    }

	VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
	build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build_info.geometryCount = uint32_t(geometries.size());
	build_info.pGeometries = geometries.data();
	// for now
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = VK_NULL_HANDLE;
	build_info.scratchData.deviceAddress = 0;

	// get the needed sizes for the buffers
	VkAccelerationStructureBuildSizesInfoKHR sizes = {};
	sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(m_device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&build_info, max_primitive_counts.data(), &sizes);

	fprintf(stdout, "BOTTOM AS: needed scratch memory %" PRIu64 " MB\n", sizes.buildScratchSize / 1024 / 1024);
	fprintf(stdout, "BOTTOM AS: needed structure memory %" PRIu64 " MB\n", sizes.accelerationStructureSize / 1024 / 1024);
	
	// create all the necessary buffers
	// structure buffer
	create_buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_bottom_as.structure_buffer);
	// scratch buffer
	create_buffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_bottom_as.scratch_buffer);

	VkAccelerationStructureCreateInfoKHR ci = {};
	ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	ci.buffer = m_bottom_as.structure_buffer.buffer;
	ci.offset = 0;
	ci.size = sizes.accelerationStructureSize;
	ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

	auto res = vkCreateAccelerationStructureKHR(m_device, &ci, nullptr, &m_bottom_as.structure);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create acceleration structure");
	
	// build as
	// fill all the addresses needed
    for (auto &geom : geometries) {
        VkAccelerationStructureGeometryTrianglesDataKHR &geom_trias = geom.geometry.triangles;
        geom_trias.vertexData.deviceAddress = vk_helpers::get_buffer_address(m_device, m_vertex_buffer.buffer);
        geom_trias.indexData.deviceAddress = vk_helpers::get_buffer_address(m_device, m_index_buffer.buffer);
        geom_trias.transformData.deviceAddress = 0;
    }
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = m_bottom_as.structure;
	build_info.scratchData.deviceAddress = vk_helpers::get_buffer_address(m_device, m_bottom_as.scratch_buffer.buffer);

    // fill all geometry build ranges
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> geom_ranges;
    for (const ModelPart &part : m_model_parts) {
        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.firstVertex = part.vertex_offset;
        range.primitiveCount = part.index_count/3;
        range.primitiveOffset = part.index_offset*sizeof(uint32_t);
        range.transformOffset = 0;
        geom_ranges.push_back(range);
    }
	const VkAccelerationStructureBuildRangeInfoKHR* p_build_ranges[] = { geom_ranges.data() };

	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);
	vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, p_build_ranges);
	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::create_bottom_acceleration_structure_spheres()
{
	VkAccelerationStructureGeometryKHR geom = {};
	geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;

	VkAccelerationStructureGeometryAabbsDataKHR& geom_aabbs = geom.geometry.aabbs;
	geom_aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
	geom_aabbs.stride = sizeof(SpherePrimitive);
	// for now 
	geom_aabbs.data.deviceAddress = 0;

	VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
	build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build_info.geometryCount = 1;
	build_info.pGeometries = &geom;
	// for now
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = VK_NULL_HANDLE;
	build_info.scratchData.deviceAddress = 0;

	const uint32_t max_primitive_counts[1] = { uint32_t(m_sphere_primitives.size()) };

	// get the needed sizes for the buffers
	VkAccelerationStructureBuildSizesInfoKHR sizes = {};
	sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(m_device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&build_info, max_primitive_counts, &sizes);

	fprintf(stdout, "BOTTOM AS SPHERES: needed scratch memory %" PRIu64 " MB\n", sizes.buildScratchSize / 1024 / 1024);
	fprintf(stdout, "BOTTOM AS SPHERES: needed structure memory %" PRIu64 " MB\n", sizes.accelerationStructureSize / 1024 / 1024);

	// create all the necessary buffers
	// structure buffer
	create_buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_bottom_as_spheres.structure_buffer);
	// scratch buffer
	create_buffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_bottom_as_spheres.scratch_buffer);

	VkAccelerationStructureCreateInfoKHR ci = {};
	ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	ci.buffer = m_bottom_as_spheres.structure_buffer.buffer;
	ci.offset = 0;
	ci.size = sizes.accelerationStructureSize;
	ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	auto res = vkCreateAccelerationStructureKHR(m_device, &ci, nullptr, &m_bottom_as_spheres.structure);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create acceleration structure");

	// build as
	// fill all the addresses needed
	geom_aabbs.data.deviceAddress = vk_helpers::get_buffer_address(m_device, m_sphere_buffer.buffer) + offsetof(SpherePrimitive, bbox);
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = m_bottom_as_spheres.structure;
	build_info.scratchData.deviceAddress = vk_helpers::get_buffer_address(m_device, m_bottom_as_spheres.scratch_buffer.buffer);

	VkAccelerationStructureBuildRangeInfoKHR geom_range = {};
	geom_range.firstVertex = 0;
	geom_range.primitiveCount = max_primitive_counts[0];
	geom_range.primitiveOffset = 0;
	geom_range.transformOffset = 0;

	VkAccelerationStructureBuildRangeInfoKHR build_ranges[] = { geom_range };
	const VkAccelerationStructureBuildRangeInfoKHR* p_build_ranges[] = { build_ranges };

	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);
	vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, p_build_ranges);
	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::create_top_acceleration_structure()
{
	VkAccelerationStructureGeometryKHR geom = {};
	geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;

	VkAccelerationStructureGeometryInstancesDataKHR& geom_instances = geom.geometry.instances;
	geom_instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geom_instances.arrayOfPointers = VK_FALSE;
	// for now 
	geom_instances.data.deviceAddress = 0;

	VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
	build_info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	build_info.geometryCount = 1;
	build_info.pGeometries = &geom;
	// for now
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = VK_NULL_HANDLE;
	build_info.scratchData.deviceAddress = 0;

	const uint32_t max_primitive_counts[1] = { 2 };

	// get the needed sizes for the buffers
	VkAccelerationStructureBuildSizesInfoKHR sizes = {};
	sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	vkGetAccelerationStructureBuildSizesKHR(m_device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&build_info, max_primitive_counts, &sizes);

	fprintf(stdout, "TOP AS: needed scratch memory %" PRIu64 " MB\n", sizes.buildScratchSize / 1024 / 1024);
	fprintf(stdout, "TOP AS: needed structure memory %" PRIu64 " MB\n", sizes.accelerationStructureSize / 1024 / 1024);

	// create all the necessary buffers
	// structure buffer
	create_buffer(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_top_as.structure_buffer);
	// scratch buffer
	create_buffer(sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_top_as.scratch_buffer);

	VkAccelerationStructureCreateInfoKHR ci = {};
	ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	ci.buffer = m_top_as.structure_buffer.buffer;
	ci.offset = 0;
	ci.size = sizes.accelerationStructureSize;
	ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	auto res = vkCreateAccelerationStructureKHR(m_device, &ci, nullptr, &m_top_as.structure);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create acceleration structure");

	// create instances buffer
	{
		VkDeviceSize instances_size = max_primitive_counts[0] * sizeof(VkAccelerationStructureInstanceKHR);
		VmaBufferAllocation staging;
		create_buffer(instances_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			staging);

		VkAccelerationStructureInstanceKHR* instance_ptr;
		auto res = vmaMapMemory(m_allocator, staging.alloc, (void**)&instance_ptr);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
		
		{
			// model
			glm::mat4 transform = m_model_tranformation;
			transform = glm::transpose(transform);
			memcpy(&instance_ptr->transform, &transform[0][0], sizeof(float) * 12);
			instance_ptr->instanceCustomIndex = 0;
			instance_ptr->mask = 0xFF;
			instance_ptr->flags = 0;
			instance_ptr->instanceShaderBindingTableRecordOffset = 0;
			instance_ptr->accelerationStructureReference = vk_helpers::get_acceleration_structure_address(m_device, m_bottom_as.structure);
		}
		instance_ptr++;
		{
			// spheres
			glm::mat4 transform = glm::mat4(1.0f);
			transform = glm::transpose(transform);
			memcpy(&instance_ptr->transform, &transform[0][0], sizeof(float) * 12);
			instance_ptr->instanceCustomIndex = 1;
			instance_ptr->mask = 0xFF;
			instance_ptr->flags = 0;
			instance_ptr->instanceShaderBindingTableRecordOffset = m_model_parts.size()*2; // here we set 2 because we have shade/shadow shaders for the first instance
			instance_ptr->accelerationStructureReference = vk_helpers::get_acceleration_structure_address(m_device, m_bottom_as_spheres.structure);;
		}

		vmaUnmapMemory(m_allocator, staging.alloc);

		create_buffer(instances_size,
			VK_BUFFER_USAGE_TRANSFER_DST_BIT |
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_top_as.instances_buffer);
		copy_buffer(staging.buffer, m_top_as.instances_buffer.buffer, instances_size);

		vmaDestroyBuffer(m_allocator, staging.buffer, staging.alloc);
	}

	// build as
	// fill all the addresses needed
	geom_instances.data.deviceAddress = vk_helpers::get_buffer_address(m_device, m_top_as.instances_buffer.buffer);
	build_info.srcAccelerationStructure = VK_NULL_HANDLE;
	build_info.dstAccelerationStructure = m_top_as.structure;
	build_info.scratchData.deviceAddress = vk_helpers::get_buffer_address(m_device, m_top_as.scratch_buffer.buffer);

	VkAccelerationStructureBuildRangeInfoKHR geom_range = {};
	geom_range.firstVertex = 0;
	geom_range.primitiveCount = max_primitive_counts[0];
	geom_range.primitiveOffset = 0;
	geom_range.transformOffset = 0;

	VkAccelerationStructureBuildRangeInfoKHR build_ranges[] = { geom_range };
	const VkAccelerationStructureBuildRangeInfoKHR* p_build_ranges[] = { build_ranges };

	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);
	vkCmdBuildAccelerationStructuresKHR(cmd_buf, 1, &build_info, p_build_ranges);
	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::create_raytracing_pipeline_layout()
{
	VkDescriptorSetLayoutBinding lb_0 = {};
	lb_0.binding = 0;
	lb_0.descriptorCount = 1;
	lb_0.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	lb_0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	lb_0.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding lb_1 = {};
	lb_1.binding = 1;
	lb_1.descriptorCount = 1;
	lb_1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	lb_1.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	lb_1.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding lb_2;
	lb_2.binding = 2;
	lb_2.descriptorCount = 1;
	lb_2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	lb_2.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	lb_2.pImmutableSamplers = nullptr;

	std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
		lb_0, lb_1, lb_2
	};

	VkDescriptorSetLayoutCreateInfo sli = {};
	sli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	sli.bindingCount = uint32_t(bindings.size());
	sli.pBindings = bindings.data();
	sli.flags = 0;
	auto res = vkCreateDescriptorSetLayout(m_device, &sli, nullptr, &m_rt_descriptor_set_layout);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create descriptor set layout");

	// Pipeline Layout
	VkPipelineLayoutCreateInfo plci = {};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	plci.flags = 0;
	plci.setLayoutCount = 1;
	plci.pSetLayouts = &m_rt_descriptor_set_layout;
	plci.pushConstantRangeCount = 0;
	plci.pPushConstantRanges = nullptr;

	res = vkCreatePipelineLayout(m_device, &plci, nullptr, &m_rt_pipeline_layout);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create pipeline layout");
	}
}

void BaseApplication::create_raytracing_pipeline()
{
	// raygen
	auto raygen_module = create_shader_module("simple.rgen", shaderc_raygen_shader, read_file(SHADER_DIR "simple.rgen"));
	auto chit_module = create_shader_module("simple.rchit", shaderc_closesthit_shader, read_file(SHADER_DIR "simple.rchit"));
	auto miss_module = create_shader_module("simple.rmiss", shaderc_miss_shader, read_file(SHADER_DIR "simple.rmiss"));
	auto shadow_chit_module = create_shader_module("shadow.rchit", shaderc_closesthit_shader, read_file(SHADER_DIR "shadow.rchit"));
	auto shadow_miss_module = create_shader_module("shadow.rmiss", shaderc_miss_shader, read_file(SHADER_DIR "shadow.rmiss"));
	auto sphere_int_module = create_shader_module("sphere.rint", shaderc_intersection_shader, read_file(SHADER_DIR "sphere.rint"));
	auto sphere_chit_module = create_shader_module("sphere.rchit", shaderc_closesthit_shader, read_file(SHADER_DIR "sphere.rchit"));

	VkPipelineShaderStageCreateInfo rgci = {};
	rgci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	rgci.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	rgci.module = raygen_module;
	rgci.pName = "main";

	VkPipelineShaderStageCreateInfo chci = {};
	chci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	chci.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	chci.module = chit_module;
	chci.pName = "main";

	VkPipelineShaderStageCreateInfo mci = {};
	mci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	mci.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	mci.module = miss_module;
	mci.pName = "main";

	VkPipelineShaderStageCreateInfo shadow_chci = {};
	shadow_chci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadow_chci.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	shadow_chci.module = shadow_chit_module;
	shadow_chci.pName = "main";

	VkPipelineShaderStageCreateInfo shadow_mci = {};
	shadow_mci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadow_mci.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	shadow_mci.module = shadow_miss_module;
	shadow_mci.pName = "main";

	VkPipelineShaderStageCreateInfo sphere_ici = {};
	sphere_ici.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	sphere_ici.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	sphere_ici.module = sphere_int_module;
	sphere_ici.pName = "main";

	VkPipelineShaderStageCreateInfo sphere_chci = {};
	sphere_chci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	sphere_chci.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	sphere_chci.module = sphere_chit_module;
	sphere_chci.pName = "main";

	std::array<VkPipelineShaderStageCreateInfo, 7> stages = { rgci, chci, mci, shadow_chci, shadow_mci, sphere_ici, sphere_chci };
	std::array<VkRayTracingShaderGroupCreateInfoKHR, 7> groups = {};

	// raygen group
	groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[0].generalShader = 0;
	groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
	groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;
	// hitgroup
	groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	groups[1].generalShader = VK_SHADER_UNUSED_KHR;
	groups[1].closestHitShader = 1;
	groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;
	// shadow hitgroup
	groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	groups[2].generalShader = VK_SHADER_UNUSED_KHR;
	groups[2].closestHitShader = 3;
	groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;
	// sphere hit group
	groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
	groups[3].generalShader = VK_SHADER_UNUSED_KHR;
	groups[3].closestHitShader = 6;
	groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[3].intersectionShader = 5;
	// sphere shadow hitgroup
	groups[4].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
	groups[4].generalShader = VK_SHADER_UNUSED_KHR;
	groups[4].closestHitShader = 3;
	groups[4].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[4].intersectionShader = 5;
	// miss group
	groups[5].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[5].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[5].generalShader = 2;
	groups[5].closestHitShader = VK_SHADER_UNUSED_KHR;
	groups[5].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[5].intersectionShader = VK_SHADER_UNUSED_KHR;
	// shadow miss group
	groups[6].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	groups[6].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	groups[6].generalShader = 4;
	groups[6].closestHitShader = VK_SHADER_UNUSED_KHR;
	groups[6].anyHitShader = VK_SHADER_UNUSED_KHR;
	groups[6].intersectionShader = VK_SHADER_UNUSED_KHR;

	VkPipelineLibraryCreateInfoKHR libci = {};
	libci.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
	libci.libraryCount = 0;

	VkRayTracingPipelineCreateInfoKHR ci = {};
	ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	ci.flags = 0;
	ci.stageCount = uint32_t(stages.size());
	ci.pStages = stages.data();
	ci.groupCount = uint32_t(groups.size());
	ci.pGroups = groups.data();
	ci.maxPipelineRayRecursionDepth = 15;
	ci.pLibraryInfo = &libci;
	ci.pLibraryInterface = nullptr;
	ci.layout = m_rt_pipeline_layout;
	ci.basePipelineHandle = VK_NULL_HANDLE;
	ci.basePipelineIndex = 0;

	auto res = vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, nullptr, &m_rt_pipeline);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create a raytracing pipeline");

	vkDestroyShaderModule(m_device, raygen_module, nullptr);
	vkDestroyShaderModule(m_device, chit_module, nullptr);
	vkDestroyShaderModule(m_device, miss_module, nullptr);
	vkDestroyShaderModule(m_device, shadow_chit_module, nullptr);
	vkDestroyShaderModule(m_device, shadow_miss_module, nullptr);
	vkDestroyShaderModule(m_device, sphere_chit_module, nullptr);
	vkDestroyShaderModule(m_device, sphere_int_module, nullptr);
}

void BaseApplication::create_rt_image()
{
	create_image(m_swapchain_extent.width, m_swapchain_extent.height, VK_FORMAT_R8G8B8A8_UNORM,
				 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_rt_img);

	m_rt_img_view = vk_helpers::create_image_view_2d(m_device, m_rt_img.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}

void BaseApplication::create_descriptor_pool()
{
	uint32_t imgs_count = (uint32_t)m_swapchain_images.size();
	// specify bigger sizes than needed
	std::array<VkDescriptorPoolSize, 3> ps = {};
	ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	ps[0].descriptorCount = 2*imgs_count;
	ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	ps[1].descriptorCount = 2*imgs_count;
	ps[2].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	ps[2].descriptorCount = 2*imgs_count;

	VkDescriptorPoolCreateInfo pi = {};
	pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pi.poolSizeCount = uint32_t(ps.size());
	pi.pPoolSizes = ps.data();
	pi.maxSets = 2*imgs_count; // one for rt and one for default

	auto res = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_desc_pool);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create descriptor pool");
}

void BaseApplication::create_descriptor_sets()
{
	std::vector<VkDescriptorSetLayout> layouts(m_swapchain_images.size(), m_descriptor_set_layout);
	
	VkDescriptorSetAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool = m_desc_pool;
	ai.descriptorSetCount = uint32_t(m_swapchain_images.size());
	ai.pSetLayouts = layouts.data();

	m_desc_sets.resize(m_swapchain_images.size());
	auto res = vkAllocateDescriptorSets(m_device, &ai, m_desc_sets.data());
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate descriptor sets");

	for (size_t i = 0; i < m_desc_sets.size(); ++i) {
		VkDescriptorBufferInfo bi = {};
		bi.buffer = m_uni_buffers[i].buffer;
		bi.offset = 0;
		bi.range = sizeof(SceneUniforms);

        VkWriteDescriptorSetAccelerationStructureKHR dw_rt = {};
		dw_rt.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		dw_rt.accelerationStructureCount = 1;
		dw_rt.pAccelerationStructures = &m_top_as.structure;

		std::array<VkWriteDescriptorSet, 2> dw = {};
		dw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[0].dstSet = m_desc_sets[i];
		dw[0].dstBinding = 0;
		dw[0].dstArrayElement = 0;
		dw[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		dw[0].descriptorCount = 1;
		dw[0].pBufferInfo = &bi;

        dw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[1].dstSet = m_desc_sets[i];
		dw[1].dstBinding = 1;
		dw[1].dstArrayElement = 0;
		dw[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		dw[1].descriptorCount = 1;
		dw[1].pNext = &dw_rt;
		
		vkUpdateDescriptorSets(m_device, uint32_t(dw.size()), dw.data(), 0, nullptr);
	}
}

void BaseApplication::create_rt_descriptor_sets()
{
	std::vector<VkDescriptorSetLayout> layouts(m_swapchain_images.size(), m_rt_descriptor_set_layout);

	VkDescriptorSetAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	ai.descriptorPool = m_desc_pool;
	ai.descriptorSetCount = uint32_t(m_swapchain_images.size());
	ai.pSetLayouts = layouts.data();

	m_rt_desc_sets.resize(m_swapchain_images.size());
	auto res = vkAllocateDescriptorSets(m_device, &ai, m_rt_desc_sets.data());
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate descriptor sets");

	for (size_t i = 0; i < m_rt_desc_sets.size(); ++i) {
		VkDescriptorImageInfo ii = {};
		ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		ii.imageView = m_rt_img_view;
		ii.sampler = nullptr;
		
		VkWriteDescriptorSetAccelerationStructureKHR dw_rt = {};
		dw_rt.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		dw_rt.accelerationStructureCount = 1;
		dw_rt.pAccelerationStructures = &m_top_as.structure;

		VkDescriptorBufferInfo ubi = {};
		ubi.buffer = m_uni_buffers[i].buffer;
		ubi.offset = 0;
		ubi.range = sizeof(SceneUniforms);

		std::array<VkWriteDescriptorSet, 3> dw = {};

		dw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[0].dstSet = m_rt_desc_sets[i];
		dw[0].dstBinding = 0;
		dw[0].dstArrayElement = 0;
		dw[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		dw[0].descriptorCount = 1;
		dw[0].pNext = &dw_rt;

		dw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[1].dstSet = m_rt_desc_sets[i];
		dw[1].dstBinding = 1;
		dw[1].dstArrayElement = 0;
		dw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		dw[1].descriptorCount = 1;
		dw[1].pImageInfo = &ii;

		dw[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[2].dstSet = m_rt_desc_sets[i];
		dw[2].dstBinding = 2;
		dw[2].dstArrayElement = 0;
		dw[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		dw[2].descriptorCount = 1;
		dw[2].pBufferInfo = &ubi;
   
		vkUpdateDescriptorSets(m_device, uint32_t(dw.size()), dw.data(), 0, nullptr);
	}

}

void BaseApplication::create_shader_binding_table()
{
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR props = vk_helpers::get_raytracing_properties(m_gpu);
	
	fprintf(stdout, "group handle size %u\n", props.shaderGroupHandleSize);
	fprintf(stdout, "group base alignment %u\n", props.shaderGroupBaseAlignment);
	fprintf(stdout, "group max stride %u\n", props.maxShaderGroupStride);
	fprintf(stdout, "max recursion depth %u\n", props.maxRayRecursionDepth);
	
	const uint32_t num_raygen = 1;
    const uint32_t num_triangle_geometries = m_model_parts.size();
    const uint32_t num_sphere_geometries = 1;
    const uint32_t num_ray_classes = 2; // shade/shadow
	const uint32_t num_hitgroups = (num_triangle_geometries+num_sphere_geometries) * num_ray_classes;
	const uint32_t num_miss = num_ray_classes;
	VkDeviceSize sz =
		num_raygen * get_sbt_raygen_record_size() +
		num_hitgroups * get_sbt_hit_record_size() +
		num_miss * get_sbt_miss_record_size();

	create_buffer(sz, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, m_rt_sbt);

	m_rt_sbt_address = vk_helpers::get_buffer_address(m_device, m_rt_sbt.buffer);

	uint8_t *data;
	auto res = vmaMapMemory(m_allocator, m_rt_sbt.alloc, (void**)&data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");

	if (props.shaderGroupHandleSize != sizeof(ShaderGroupHandle)) {
		throw std::runtime_error("we assume at compile time that shadergroup handle size is 32 bytes");
	}
	if (props.shaderGroupBaseAlignment != 64) {
		throw std::runtime_error("we assume at compile time that shadergroup base alignment is 64 bytes");
	}

	const uint32_t group_count = 7u;
	ShaderGroupHandle handles[group_count];
	vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rt_pipeline, 0, group_count, sizeof(ShaderGroupHandle) * group_count, handles);

	// write raygen groups
	{
		ShaderGroupHandle raygen_rec = handles[0];
		std::memcpy(data, &raygen_rec, sizeof(raygen_rec));
		data += get_sbt_raygen_record_size();
	}
	// write hitgroups
	{
		// hit groups
		// triangles 
        for (const ModelPart &part : m_model_parts) {
            SBTRecordHitMesh mesh_rec;
            mesh_rec.shader = handles[1];
            mesh_rec.vertices_ref = sizeof(Vertex)*part.vertex_offset +
                vk_helpers::get_buffer_address(m_device, m_vertex_buffer.buffer);
            mesh_rec.indices_ref = sizeof(uint32_t)*part.index_offset + 
                vk_helpers::get_buffer_address(m_device, m_index_buffer.buffer);
			mesh_rec.pbr_material = part.pbr_material;

            ShaderGroupHandle mesh_occlusion_rec = handles[2];

            std::memcpy(data, &mesh_rec, sizeof(mesh_rec));
            data += get_sbt_hit_record_size();
            // triangles shadow
            std::memcpy(data, &mesh_occlusion_rec, sizeof(mesh_occlusion_rec));
            data += get_sbt_hit_record_size();
        }
        {
            SBTRecordHitSphere spheres_rec;
            spheres_rec.shader = handles[3];
            spheres_rec.spheres_ref = vk_helpers::get_buffer_address(m_device, m_sphere_buffer.buffer);

            ShaderGroupHandle spheres_occlusion_rec = handles[4];

            // spheres
            std::memcpy(data, &spheres_rec, sizeof(spheres_rec));
            data += get_sbt_hit_record_size();
            // spheres shadow
            std::memcpy(data, &spheres_occlusion_rec, sizeof(spheres_occlusion_rec));
            data += get_sbt_hit_record_size();
        }
	}
	// write miss groups
	{
		ShaderGroupHandle miss_rec = handles[5];
		ShaderGroupHandle miss_occlusion_rec = handles[6];
		// miss groups 
		std::memcpy(data, &miss_rec, sizeof(miss_rec));
		data += get_sbt_miss_record_size();
		// miss shadow
		std::memcpy(data, &miss_occlusion_rec, sizeof(miss_occlusion_rec));
	}
	

	vmaUnmapMemory(m_allocator, m_rt_sbt.alloc);
}

VkFormat BaseApplication::find_supported_format(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const
{
	for (auto f : candidates) {
		VkFormatProperties props;
		vkGetPhysicalDeviceFormatProperties(m_gpu, f, &props);
		if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
			return f;
		} else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
			return f;
		}
	}

	throw std::runtime_error("failed to find supported format");
}

VkFormat BaseApplication::find_supported_depth_format() const
{
	std::vector<VkFormat> candidates = {
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT
	};
	return find_supported_format(candidates, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void BaseApplication::create_depth_resources()
{
	VkFormat depth_format = find_supported_depth_format();
	create_image(m_swapchain_extent.width, m_swapchain_extent.height, depth_format,
				 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depth_img);

	m_depth_img_view = vk_helpers::create_image_view_2d(m_device, m_depth_img.image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void BaseApplication::create_command_pools()
{
	auto indices = find_queue_families(m_gpu);
	VkCommandPoolCreateInfo pci = {};
	pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pci.queueFamilyIndex = indices.graphics_family.value();
	pci.flags = 0; // optional

	auto res = vkCreateCommandPool(m_device, &pci, nullptr, &m_graphics_cmd_pool);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool");
	}

	pci.queueFamilyIndex = indices.transfer_family.value();
	res = vkCreateCommandPool(m_device, &pci, nullptr, &m_transfer_cmd_pool);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool");
	}
}

void BaseApplication::create_command_buffers()
{
	// the command buffers are the same number as the fbs;
	m_cmd_buffers.resize(m_swapchain_fbs.size());

	VkCommandBufferAllocateInfo cbi = {};
	cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbi.commandPool = m_graphics_cmd_pool;
	cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = (uint32_t)m_cmd_buffers.size();

	auto res = vkAllocateCommandBuffers(m_device, &cbi, m_cmd_buffers.data());
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers");
	}
	
	for (size_t i = 0; i < m_cmd_buffers.size(); ++i) {
		VkCommandBufferBeginInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = 0;
		bi.pInheritanceInfo = nullptr;
		res = vkBeginCommandBuffer(m_cmd_buffers[i], &bi);
		if (res != VK_SUCCESS) { throw std::runtime_error("failed to begin recording commands"); }
		
		VkRenderPassBeginInfo rpbi = {};
		rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		rpbi.renderPass = m_render_pass;
		rpbi.framebuffer = m_swapchain_fbs[i];
		rpbi.renderArea.offset = { 0, 0 };
		rpbi.renderArea.extent = m_swapchain_extent;
		std::array<VkClearValue, 2> clear_values = {};
		clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clear_values[1].depthStencil = { 1.0f, 0 };
		rpbi.clearValueCount = uint32_t(clear_values.size());
		rpbi.pClearValues = clear_values.data();

		vkCmdBeginRenderPass(m_cmd_buffers[i], &rpbi, VK_SUBPASS_CONTENTS_INLINE);
	
		vkCmdBindPipeline(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphics_pipeline);
		
		VkBuffer buffers[] = { m_vertex_buffer.buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(m_cmd_buffers[i], 0, 1, buffers, offsets);
		vkCmdBindIndexBuffer(m_cmd_buffers[i], m_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout,
								0, 1, &m_desc_sets[i], 0, nullptr);
        for (auto p : m_model_parts) {
            vkCmdDrawIndexed(m_cmd_buffers[i], p.index_count, 1, p.index_offset, p.vertex_offset, 0);
        }

		vkCmdEndRenderPass(m_cmd_buffers[i]);

		res = vkEndCommandBuffer(m_cmd_buffers[i]);
		if (res != VK_SUCCESS) {
			throw std::runtime_error("failed to end recording commands");
		}
	}
}

void BaseApplication::create_rt_command_buffers()
{
	// the command buffers are the same number as the fbs;
	m_rt_cmd_buffers.resize(m_swapchain_fbs.size());

	VkCommandBufferAllocateInfo cbi = {};
	cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	cbi.commandPool = m_graphics_cmd_pool;
	cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = (uint32_t)m_rt_cmd_buffers.size();

	auto res = vkAllocateCommandBuffers(m_device, &cbi, m_rt_cmd_buffers.data());
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to allocate command buffers");
	}

	for (size_t i = 0; i < m_rt_cmd_buffers.size(); ++i) {
		VkCommandBufferBeginInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		bi.flags = 0;
		bi.pInheritanceInfo = nullptr;
		res = vkBeginCommandBuffer(m_rt_cmd_buffers[i], &bi);
		if (res != VK_SUCCESS) { throw std::runtime_error("failed to begin recording commands"); }

		VkImageSubresourceRange isr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
		vk_helpers::image_barrier(m_rt_cmd_buffers[i], m_rt_img.image, isr,
			VK_PIPELINE_STAGE_2_NONE_KHR, VK_ACCESS_2_NONE_KHR, VK_IMAGE_LAYOUT_UNDEFINED,
			VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_GENERAL);
	
		vkCmdBindPipeline(m_rt_cmd_buffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rt_pipeline);
		vkCmdBindDescriptorSets(m_rt_cmd_buffers[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rt_pipeline_layout,
			0, 1, &m_rt_desc_sets[i], 0, nullptr);

		const size_t raygen_stride = get_sbt_raygen_record_size();
		const size_t hitgroup_stride = get_sbt_hit_record_size();
		const size_t miss_stride = get_sbt_miss_record_size();
        const uint32_t num_raygen = 1;
        const uint32_t num_triangle_geometries = m_model_parts.size();
        const uint32_t num_sphere_geometries = 1;
        const uint32_t num_ray_classes = 2; // shade/shadow
        const uint32_t num_hitgroups = (num_triangle_geometries+num_sphere_geometries) * num_ray_classes;
        const uint32_t num_miss = num_ray_classes;
		size_t raygen_offset = 0;
		size_t hitgroups_offset = raygen_stride * num_raygen;
		size_t miss_offset = hitgroups_offset + hitgroup_stride * num_hitgroups;
		VkStridedDeviceAddressRegionKHR raygen_region = { 
			m_rt_sbt_address+raygen_offset, 
			raygen_stride, 
			raygen_stride*num_raygen 
		};
		VkStridedDeviceAddressRegionKHR hitgroup_region = { 
			m_rt_sbt_address+hitgroups_offset, 
			hitgroup_stride, 
			hitgroup_stride*num_hitgroups 
		};
		VkStridedDeviceAddressRegionKHR miss_region = { 
			m_rt_sbt_address+miss_offset, 
			miss_stride, 
			miss_stride*num_miss 
		};
		VkStridedDeviceAddressRegionKHR callable_region = { 0, 0, 0 };
		vkCmdTraceRaysKHR(m_rt_cmd_buffers[i],
			&raygen_region, &miss_region, &hitgroup_region, &callable_region,
			m_width, m_height, 1);

		vk_helpers::image_barrier(m_rt_cmd_buffers[i], m_rt_img.image, isr,
			VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_GENERAL,
			VK_PIPELINE_STAGE_2_BLIT_BIT_KHR, VK_ACCESS_2_TRANSFER_READ_BIT_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		
		vk_helpers::image_barrier(m_rt_cmd_buffers[i], m_swapchain_images[i], isr,
			VK_PIPELINE_STAGE_2_NONE_KHR, VK_ACCESS_2_NONE_KHR, VK_IMAGE_LAYOUT_UNDEFINED,
			VK_PIPELINE_STAGE_2_BLIT_BIT_KHR, VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        VkImageBlit blit = {};
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { int32_t(m_width), int32_t(m_height), 1 };
        blit.srcOffsets[0] = { 0, 0, 0 };
        blit.srcOffsets[1] = { int32_t(m_width), int32_t(m_height), 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        vkCmdBlitImage(m_rt_cmd_buffers[i],
                m_rt_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                m_swapchain_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

		vk_helpers::image_barrier(m_rt_cmd_buffers[i], m_swapchain_images[i], isr,
			VK_PIPELINE_STAGE_2_BLIT_BIT_KHR, VK_ACCESS_2_TRANSFER_WRITE_BIT_KHR, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			VK_PIPELINE_STAGE_2_NONE_KHR, VK_ACCESS_2_NONE_KHR, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
	
		res = vkEndCommandBuffer(m_rt_cmd_buffers[i]);
		if (res != VK_SUCCESS) {
			throw std::runtime_error("failed to end recording commands");
		}
	}
}

void BaseApplication::create_sync_objects()
{
	VkSemaphoreCreateInfo sci = {};
	sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fci = {};
	fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fci.flags = VK_FENCE_CREATE_SIGNALED_BIT; // first time needs to be signaled

	std::array<VkResult, MAX_FRAMES_IN_FLIGHT> res;
	auto pred = [](VkResult r) {return r == VK_SUCCESS; };

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		res[i] = vkCreateSemaphore(m_device, &sci, nullptr, &m_sem_img_available[i]);
	}
	if (!std::all_of(res.begin(), res.end(), pred)) 
		throw std::runtime_error("failed to create semaphores");

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		res[i] = vkCreateSemaphore(m_device, &sci, nullptr, &m_sem_render_finished[i]);
	}
	if (!std::all_of(res.begin(), res.end(), pred))
		throw std::runtime_error("failed to create semaphores");
	
	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		res[i] = vkCreateFence(m_device, &fci, nullptr, &m_fen_flight[i]);
	}
	if (!std::all_of(res.begin(), res.end(), pred))
		throw std::runtime_error("failed to create fences");
}

void BaseApplication::update_uniform_buffer(uint32_t idx)
{
	static auto start_time = std::chrono::high_resolution_clock::now();
	//auto curr_time = std::chrono::high_resolution_clock::now();
	//float time = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(start_time.time_since_epoch()).count();

	SceneUniforms ubo = {};
	ubo.model = m_model_tranformation;
	//ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = m_camera.get_view_matrix();
	ubo.proj = glm::perspective(glm::radians(45.0f), m_swapchain_extent.width / (float)m_swapchain_extent.height, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;
	ubo.iview = glm::inverse(ubo.view);
	ubo.iproj = glm::inverse(ubo.proj);
	ubo.samples_accum = (m_samples_accumulated++);

	ubo.light_pos = glm::vec4(4.0f * std::cos(time), 4.0f * std::sin(time), 5.0f, 1.0f);
	void *data;
	auto res = vmaMapMemory(m_allocator, m_uni_buffers[idx].alloc, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map uniform buffer memory");
	std::memcpy(data, &ubo, sizeof(SceneUniforms));
	vmaUnmapMemory(m_allocator, m_uni_buffers[idx].alloc);
}

void BaseApplication::draw_frame()
{
	vkWaitForFences(m_device, 1, &m_fen_flight[m_current_frame_idx], 
					VK_TRUE, std::numeric_limits<uint64_t>::max());
	
	uint32_t img_idx;
	auto res = vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<uint64_t>::max(),
						  m_sem_img_available[m_current_frame_idx], 
						  VK_NULL_HANDLE, &img_idx);

	if (res == VK_ERROR_OUT_OF_DATE_KHR) {
		recreate_swapchain(); 
		return;
	} else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("failed to acquire swapchain image");
	}

	update_uniform_buffer(img_idx);

	VkSemaphoreSubmitInfoKHR wait_sem = {};
	wait_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR;
	wait_sem.semaphore = m_sem_img_available[m_current_frame_idx];
	if (m_raytraced) {
		wait_sem.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;
	} else {
		wait_sem.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
	}
	wait_sem.deviceIndex = 0;

	// Waits until everything is done
	VkSemaphoreSubmitInfoKHR signal_sem = {};
	signal_sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR;
	signal_sem.semaphore = m_sem_render_finished[m_current_frame_idx];
	signal_sem.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
	signal_sem.deviceIndex = 0;

	VkCommandBufferSubmitInfoKHR cmd_submit = {};
	cmd_submit.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR;
	cmd_submit.commandBuffer = m_raytraced ? m_rt_cmd_buffers[img_idx] : m_cmd_buffers[img_idx];
	cmd_submit.deviceMask = 0;
	
	VkSubmitInfo2KHR submit_info = {};
	submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR;
	submit_info.waitSemaphoreInfoCount = 1;
	submit_info.pWaitSemaphoreInfos = &wait_sem;
	submit_info.signalSemaphoreInfoCount = 1;
	submit_info.pSignalSemaphoreInfos = &signal_sem;
	submit_info.commandBufferInfoCount = 1;
	submit_info.pCommandBufferInfos = &cmd_submit;

	// we reset fences here because we need it after checking for swapchain recreation
	// else we could apply it after vkWaitForFences
	vkResetFences(m_device, 1, &m_fen_flight[m_current_frame_idx]);
	res = vkQueueSubmit2KHR(m_graphics_queue, 1, &submit_info, m_fen_flight[m_current_frame_idx]);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to submit command buffers to queue");
	}

	VkPresentInfoKHR pi = {};
	pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = &m_sem_render_finished[m_current_frame_idx];
	VkSwapchainKHR swapchains[] = { m_swapchain };
	pi.pSwapchains = swapchains;
	pi.swapchainCount = 1;
	pi.pImageIndices = &img_idx;
	pi.pResults = nullptr;

	res = vkQueuePresentKHR(m_present_queue, &pi);
	if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || m_window_resized) {
		m_window_resized = false;
		recreate_swapchain();
	} else if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to present swapchain image");
	}
	m_current_frame_idx = (m_current_frame_idx + 1) % MAX_FRAMES_IN_FLIGHT;
}

int main()
{
	BaseApplication app;
	try {
		app.run();
	} catch (const std::exception &e) {
		fprintf(stderr, "%s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}




