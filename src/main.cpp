#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE // projection matrix depth range 0-1
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <tiny_obj_loader.h>
#include "stb_image.h"

const int MAX_FRAMES_IN_FLIGHT = 3;
#define ENABLE_VALIDATION_LAYERS

struct VkGeometryInstanceNV
{
	float          transform[12];
	uint32_t       instanceCustomIndex : 24;
	uint32_t       mask : 8;
	uint32_t       instanceOffset : 24;
	uint32_t       flags : 8;
	uint64_t       accelerationStructureHandle;
};

static PFN_vkCreateAccelerationStructureNV evkCreateAccelerationStructureNV;
static PFN_vkDestroyAccelerationStructureNV evkDestroyAccelerationStructureNV;
static PFN_vkGetAccelerationStructureMemoryRequirementsNV evkGetAccelerationStructureMemoryRequirementsNV;
static PFN_vkBindAccelerationStructureMemoryNV evkBindAccelerationStructureMemoryNV;
static PFN_vkCmdBuildAccelerationStructureNV evkCmdBuildAccelerationStructureNV;
static PFN_vkGetAccelerationStructureHandleNV evkGetAccelerationStructureHandleNV;
static PFN_vkCreateRayTracingPipelinesNV evkCreateRayTracingPipelinesNV;


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
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 tex_coord;

	bool operator == (const Vertex &other) const
	{
		return pos == other.pos && color == other.color && tex_coord == other.tex_coord;
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
		ad[1].offset = offsetof(Vertex, color);
		ad[2].binding = 0;
		ad[2].location = 2;
		ad[2].format = VK_FORMAT_R32G32_SFLOAT;
		ad[2].offset = offsetof(Vertex, tex_coord);

		return ad;
	}
};

// implement has specialization for vertex
namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^
					(hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
					(hash<glm::vec2>()(vertex.tex_coord) << 1);
		}
	};
}


struct CameraMatrices
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct ASBuffers
{
	VkAccelerationStructureNV structure{ VK_NULL_HANDLE };
	VkDeviceMemory memory{ VK_NULL_HANDLE };
	VkBuffer scratch_buffer{ VK_NULL_HANDLE };
	VkDeviceMemory scratch_memory{ VK_NULL_HANDLE };
	VkBuffer instances{ VK_NULL_HANDLE };
	VkDeviceMemory instances_memory{ VK_NULL_HANDLE };

	void destroy(VkDevice device)
	{
		if (structure) evkDestroyAccelerationStructureNV(device, structure, nullptr);
		vkDestroyBuffer(device, scratch_buffer, nullptr);
		vkDestroyBuffer(device, instances, nullptr);
		vkFreeMemory(device, memory, nullptr);
		vkFreeMemory(device, scratch_memory, nullptr);
		vkFreeMemory(device, instances_memory, nullptr);
		std::memset(this, VK_NULL_HANDLE, sizeof(ASBuffers));
	}
};

class BaseApplication
{
public:
	BaseApplication();
	~BaseApplication();
	void run();
	void set_window_resized() { m_window_resized = true; }

private:
	void init_window();
	void init_vulkan();
	void main_loop();
	void cleanup();

	void create_instance();
	bool check_validation_layer_support() const;
	std::vector<const char*> get_required_extensions() const;

	void setup_debug_callback();
	void destroy_debug_callback();
	void setup_raytracing_device_functions();

	void pick_gpu();
	bool check_device_extension_support(VkPhysicalDevice gpu) const;
	bool is_gpu_suitable(VkPhysicalDevice gpu) const;
	QueueFamilyIndices find_queue_families(VkPhysicalDevice gpu) const;

	void create_logical_device();
	
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
	VkShaderModule create_shader_module(const std::vector<char> &code) const;

	void create_framebuffers();

	uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
	
	void create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, 
					   VkMemoryPropertyFlags props, VkBuffer &buffer, VkDeviceMemory &memory);
	void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
	
	void create_image(uint32_t width, uint32_t height, VkFormat format,
					  VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags props,
					  VkImage &img, VkDeviceMemory &img_memory);
	
	VkCommandBuffer begin_single_time_commands(VkQueue queue, VkCommandPool cmd_pool);
	void end_single_time_commands(VkQueue queue, VkCommandPool cmd_pool, VkCommandBuffer cmd_buffer);

	void transition_image_layout(VkImage img, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout);
	void copy_buffer_to_image(VkBuffer buffer, VkImage img, uint32_t width, uint32_t height);

	void load_model();

	void create_vertex_buffer();
	void create_index_buffer();
	void create_uniform_buffers();

	void create_bottom_acceleration_structure();
	void create_top_acceleration_structure();
	void create_raytracing_pipeline_layout();
	void create_raytracing_pipeline();

	void create_texture_image();
	void create_texture_image_view();
	void create_texture_sampler();

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
	
	void create_sync_objects();

	void update_uniform_buffer(uint32_t idx);
	void draw_frame();

	void cleanup_swapchain();
	void recreate_swapchain();

private:
	GLFWwindow *m_window{ nullptr };
	uint32_t m_width{ 1024 };
	uint32_t m_height{ 768 };

	std::vector<const char*> m_validation_layers;
	bool m_enable_validation_layers;

	std::vector<const char*> m_device_extensions;
	size_t m_current_frame_idx{ 0 };
	bool m_window_resized{ false };

	VkInstance m_instance{ VK_NULL_HANDLE };
	VkDebugUtilsMessengerEXT m_debug_callback{ VK_NULL_HANDLE };
	VkPhysicalDevice m_gpu{ VK_NULL_HANDLE };

	VkDevice m_device{ VK_NULL_HANDLE };
	VkQueue m_graphics_queue{ VK_NULL_HANDLE };
	VkQueue m_present_queue{ VK_NULL_HANDLE };
	VkQueue m_transfer_queue{ VK_NULL_HANDLE };
	
	VkSurfaceKHR m_surface{ VK_NULL_HANDLE };

	VkSwapchainKHR m_swapchain{ VK_NULL_HANDLE };
	std::vector<VkImage> m_swapchain_images;
	VkFormat m_swapchain_img_format;
	VkExtent2D m_swapchain_extent;
	std::vector<VkImageView> m_swapchain_img_views;
	std::vector<VkFramebuffer> m_swapchain_fbs;
	
	VkImage m_depth_img;
	VkDeviceMemory m_depth_img_memory;
	VkImageView m_depth_img_view;

	VkRenderPass m_render_pass{ VK_NULL_HANDLE };
	VkDescriptorSetLayout m_descriptor_set_layout{ VK_NULL_HANDLE };
	VkPipelineLayout m_pipeline_layout{ VK_NULL_HANDLE };
	VkPipeline m_graphics_pipeline{ VK_NULL_HANDLE };
	
	VkDescriptorSetLayout m_rt_descriptor_set_layout{ VK_NULL_HANDLE };
	VkPipelineLayout m_rt_pipeline_layout {VK_NULL_HANDLE};
	VkPipeline m_rt_pipeline{ VK_NULL_HANDLE };
	
	std::vector<Vertex> m_model_vertices;
	std::vector<uint32_t> m_model_indices;

	VkBuffer m_vertex_buffer{ VK_NULL_HANDLE };
	VkDeviceMemory m_vertex_buffer_memory{ VK_NULL_HANDLE };
	VkBuffer m_index_buffer{ VK_NULL_HANDLE };
	VkDeviceMemory m_index_buffer_memory{ VK_NULL_HANDLE };

	ASBuffers m_bottom_as;
	ASBuffers m_top_as;
	VkImage m_rt_img{ VK_NULL_HANDLE };
	VkDeviceMemory m_rt_img_memory{ VK_NULL_HANDLE };
	VkImageView m_rt_img_view{ VK_NULL_HANDLE };
	VkBuffer m_rt_sbt{ VK_NULL_HANDLE };
	VkDeviceMemory m_rt_sbt_memory{ VK_NULL_HANDLE };

	std::vector<VkBuffer> m_uni_buffers;
	std::vector<VkDeviceMemory> m_uni_buffer_memory;

	VkImage m_texture_img{ VK_NULL_HANDLE };
	VkDeviceMemory m_texture_img_memory{ VK_NULL_HANDLE };
	VkImageView m_texture_img_view{ VK_NULL_HANDLE };
	VkSampler m_texture_sampler{ VK_NULL_HANDLE };

	VkDescriptorPool m_desc_pool{ VK_NULL_HANDLE };
	std::vector<VkDescriptorSet> m_desc_sets;
	std::vector<VkDescriptorSet> m_rt_desc_sets;

	VkCommandPool m_graphics_cmd_pool{ VK_NULL_HANDLE };
	VkCommandPool m_transfer_cmd_pool{ VK_NULL_HANDLE };
	std::vector<VkCommandBuffer> m_cmd_buffers;
	
	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_sem_img_available{ VK_NULL_HANDLE };
	std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_sem_render_finished{ VK_NULL_HANDLE };
	std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_fen_flight{ VK_NULL_HANDLE };
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
	}
}

static void framebuffer_resize_callback(GLFWwindow *window, int width, int height)
{
	auto app = reinterpret_cast<BaseApplication*>(glfwGetWindowUserPointer(window));
	app->set_window_resized();
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

VkPhysicalDeviceRayTracingPropertiesNV get_raytracing_properties(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceProperties2 props = {};
	props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	VkPhysicalDeviceRayTracingPropertiesNV rt_props = {};
	rt_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
	props.pNext = &rt_props;
	vkGetPhysicalDeviceProperties2(gpu, &props);
	return rt_props;
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
	m_validation_layers.push_back("VK_LAYER_LUNARG_standard_validation");
#if !defined(ENABLE_VALIDATION_LAYERS)
	m_enable_validation_layers = false;
#else 
	m_enable_validation_layers = true;
#endif

	m_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	m_device_extensions.push_back(VK_NV_RAY_TRACING_EXTENSION_NAME);
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
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	m_window = glfwCreateWindow(m_width, m_height, "tutorial", NULL, NULL);
	if (!m_window) {
		throw std::runtime_error("could not create glfw window");
	}

	glfwSetKeyCallback(m_window, key_callback);
	glfwSetFramebufferSizeCallback(m_window, framebuffer_resize_callback);
	glfwSetWindowUserPointer(m_window, this);
}

void BaseApplication::init_vulkan()
{
	create_instance();
	if (m_enable_validation_layers) {
		setup_debug_callback();
	}
	setup_raytracing_device_functions();

	create_surface();

	pick_gpu();
	create_logical_device();
	
	create_command_pools();

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
	create_texture_image();
	create_texture_image_view();
	create_texture_sampler();

	create_bottom_acceleration_structure();
	create_top_acceleration_structure();

	create_descriptor_pool();
	create_descriptor_sets();
	create_rt_descriptor_sets();
	create_shader_binding_table();

	create_command_buffers();

	create_sync_objects();
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
	for (auto b : m_uni_buffers) {
		vkDestroyBuffer(m_device, b, nullptr);	
	}
	for (auto m : m_uni_buffer_memory) {
		vkFreeMemory(m_device, m, nullptr);
	}

	// no need to free desc sets because we destroy the pool
	if (m_desc_pool) vkDestroyDescriptorPool(m_device, m_desc_pool, nullptr);
	if (m_device) {
		vkFreeCommandBuffers(m_device, m_graphics_cmd_pool,
							 static_cast<uint32_t>(m_cmd_buffers.size()), m_cmd_buffers.data());
	}
	for (auto fb : m_swapchain_fbs) {
		vkDestroyFramebuffer(m_device, fb, nullptr);
	}

	if (m_graphics_pipeline) vkDestroyPipeline(m_device, m_graphics_pipeline, nullptr);
	if (m_descriptor_set_layout) vkDestroyDescriptorSetLayout(m_device, m_descriptor_set_layout, nullptr);
	if (m_pipeline_layout) vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
	if (m_render_pass) vkDestroyRenderPass(m_device, m_render_pass, nullptr);
	
	if (m_depth_img_view) vkDestroyImageView(m_device, m_depth_img_view, nullptr);
	if (m_depth_img) vkDestroyImage(m_device, m_depth_img, nullptr);
	if (m_depth_img_memory) vkFreeMemory(m_device, m_depth_img_memory, nullptr);

	// cleanup raytracing stuff
	if (m_device) {
		vkDestroyImageView(m_device, m_rt_img_view, nullptr);
		vkDestroyImage(m_device, m_rt_img, nullptr);
		vkFreeMemory(m_device, m_rt_img_memory, nullptr);
	}

	for (auto img_view : m_swapchain_img_views) {
		vkDestroyImageView(m_device, img_view, nullptr);
	}
	if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
}

void BaseApplication::cleanup()
{
	cleanup_swapchain();

	// cleanup raytracing stuff
	if (m_device) {
		vkDestroyDescriptorSetLayout(m_device, m_rt_descriptor_set_layout, nullptr);
		vkDestroyPipelineLayout(m_device, m_rt_pipeline_layout, nullptr);
		vkDestroyPipeline(m_device, m_rt_pipeline, nullptr);
		m_top_as.destroy(m_device);
		m_bottom_as.destroy(m_device);
	}

	if (m_device) {
		vkDestroySampler(m_device, m_texture_sampler, nullptr);
		vkDestroyImageView(m_device, m_texture_img_view, nullptr);
		vkDestroyImage(m_device, m_texture_img, nullptr);
		vkFreeMemory(m_device, m_texture_img_memory, nullptr);
		vkDestroyBuffer(m_device, m_index_buffer, nullptr);
		vkFreeMemory(m_device, m_index_buffer_memory, nullptr);
		vkDestroyBuffer(m_device, m_vertex_buffer, nullptr);
		vkFreeMemory(m_device, m_vertex_buffer_memory, nullptr);
	}
	

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
		vkDestroyFence(m_device, m_fen_flight[i], nullptr);
		vkDestroySemaphore(m_device, m_sem_img_available[i], nullptr);
		vkDestroySemaphore(m_device, m_sem_render_finished[i], nullptr);
	}
	if (m_transfer_cmd_pool) vkDestroyCommandPool(m_device, m_transfer_cmd_pool, nullptr);
	if (m_graphics_cmd_pool) vkDestroyCommandPool(m_device, m_graphics_cmd_pool, nullptr);
	if (m_device) vkDestroyDevice(m_device, nullptr);
	if (m_surface) vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

	if (m_debug_callback) destroy_debug_callback();
	if (m_instance) vkDestroyInstance(m_instance, nullptr);

	if (m_window) glfwDestroyWindow(m_window);
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
	ai.apiVersion = VK_API_VERSION_1_1;

	auto required_exts = get_required_extensions();

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
	ci.enabledLayerCount = 0;
	
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

std::vector<const char*> BaseApplication::get_required_extensions() const
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

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		m_instance, "vkCreateDebugUtilsMessengerEXT");
	if (func == nullptr) {
		throw std::runtime_error("debug messages are not supported");
		return;
	}
	auto res = func(m_instance, &ci, nullptr, &m_debug_callback);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to setup debug callback");
	}
}

void BaseApplication::destroy_debug_callback()
{
	if (!m_debug_callback) return;
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		m_instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func) func(m_instance, m_debug_callback, nullptr);
}

void BaseApplication::setup_raytracing_device_functions()
{
	evkCreateAccelerationStructureNV = (PFN_vkCreateAccelerationStructureNV)
		vkGetInstanceProcAddr(m_instance, "vkCreateAccelerationStructureNV");
	evkDestroyAccelerationStructureNV = (PFN_vkDestroyAccelerationStructureNV)
		vkGetInstanceProcAddr(m_instance, "vkDestroyAccelerationStructureNV");
	evkGetAccelerationStructureMemoryRequirementsNV = (PFN_vkGetAccelerationStructureMemoryRequirementsNV)
		vkGetInstanceProcAddr(m_instance, "vkGetAccelerationStructureMemoryRequirementsNV");
	evkBindAccelerationStructureMemoryNV = (PFN_vkBindAccelerationStructureMemoryNV)
		vkGetInstanceProcAddr(m_instance, "vkBindAccelerationStructureMemoryNV");
	evkCmdBuildAccelerationStructureNV = (PFN_vkCmdBuildAccelerationStructureNV)
		vkGetInstanceProcAddr(m_instance, "vkCmdBuildAccelerationStructureNV");
	evkGetAccelerationStructureHandleNV = (PFN_vkGetAccelerationStructureHandleNV)
		vkGetInstanceProcAddr(m_instance, "vkGetAccelerationStructureHandleNV");
	evkCreateRayTracingPipelinesNV = (PFN_vkCreateRayTracingPipelinesNV)
		vkGetInstanceProcAddr(m_instance, "vkCreateRayTracingPipelinesNV");
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
	VkPhysicalDeviceFeatures features;
	vkGetPhysicalDeviceFeatures(gpu, &features);
	
	auto indices = find_queue_families(gpu);

	bool extensions_supported = check_device_extension_support(gpu);
	bool swapchain_adequate = false;
	if (extensions_supported) {
		auto chain_details = query_swapchain_support(gpu);
		swapchain_adequate = !chain_details.formats.empty() && 
			!chain_details.present_modes.empty();
	}

	bool supported_features = features.vertexPipelineStoresAndAtomics && features.samplerAnisotropy;
	
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

	VkPhysicalDeviceFeatures device_features = {};
	device_features.samplerAnisotropy = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics = VK_TRUE;

	VkDeviceCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	ci.pQueueCreateInfos = qcis.data();
	ci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
	ci.pEnabledFeatures = &device_features;

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
	ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

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
	VkAttachmentDescription color_attachment = {};
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

	VkAttachmentDescription depth_attachment = {};
	depth_attachment.format = find_supported_depth_format();
	depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference color_attachment_ref = {};
	color_attachment_ref.attachment = 0; // index to above descriptions
	color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_attachment_ref = {};
	depth_attachment_ref.attachment = 1;
	depth_attachment_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	
	VkSubpassDescription subpass = {};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	// the index in this array is referenced in the frag shader
	subpass.pColorAttachments = &color_attachment_ref;
	subpass.pDepthStencilAttachment = &depth_attachment_ref;

	VkSubpassDependency dependency = {};
	dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass = 0; // our subpass
	dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.srcAccessMask = 0;
	dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

	std::array<VkAttachmentDescription, 2> attachments = {
		color_attachment, depth_attachment
	};
	VkRenderPassCreateInfo rpci = {};
	rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	rpci.attachmentCount = uint32_t(attachments.size());
	rpci.pAttachments = attachments.data();
	rpci.subpassCount = 1;
	rpci.pSubpasses = &subpass;
	rpci.dependencyCount = 1;
	rpci.pDependencies = &dependency;

	auto res = vkCreateRenderPass(m_device, &rpci, nullptr, &m_render_pass);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to create render pass");
	}
}

void BaseApplication::create_descriptor_set_layout()
{
	VkDescriptorSetLayoutBinding lb = {};
	lb.binding = 0;
	lb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	lb.descriptorCount = 1;
	lb.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
	lb.pImmutableSamplers = nullptr;

	VkDescriptorSetLayoutBinding sb = {};
	sb.binding = 1;
	sb.descriptorCount = 1;
	sb.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	sb.pImmutableSamplers = nullptr;
	sb.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
		lb, sb
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
	auto vert_code = read_file("resources/simple.vert.spv");
	auto frag_code = read_file("resources/simple.frag.spv");
	auto vert_module = create_shader_module(vert_code);
	auto frag_module = create_shader_module(frag_code);

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

VkShaderModule BaseApplication::create_shader_module(const std::vector<char>& code) const
{
	VkShaderModuleCreateInfo ci = {};
	ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	ci.codeSize = code.size();
	// cast bytes to uints
	ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
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
									VkBuffer &buffer, VkDeviceMemory &memory)
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
	bi.pQueueFamilyIndices = qidx;
	bi.queueFamilyIndexCount = 2;

	auto res = vkCreateBuffer(m_device, &bi, nullptr, &buffer);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create buffer");

	VkMemoryRequirements mem_req;
	vkGetBufferMemoryRequirements(m_device, buffer, &mem_req);
	VkMemoryAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = mem_req.size;
	ai.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, props);

	res = vkAllocateMemory(m_device, &ai, nullptr, &memory);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");

	vkBindBufferMemory(m_device, buffer, memory, 0);
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
								   VkMemoryPropertyFlags props, VkImage & img, VkDeviceMemory & img_memory)
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

	auto res = vkCreateImage(m_device, &ii, nullptr, &img);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create image");

	VkMemoryRequirements mem_req;
	vkGetImageMemoryRequirements(m_device, img, &mem_req);
	VkMemoryAllocateInfo ai = {};
	ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	ai.allocationSize = mem_req.size;
	ai.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, props);

	res = vkAllocateMemory(m_device, &ai, nullptr, &img_memory);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate memory");

	vkBindImageMemory(m_device, img, img_memory, 0);
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

	VkSubmitInfo si = {};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &cmd_buffer;

	res = vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to submit to queue");
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(m_device, cmd_pool, 1, &cmd_buffer);
}

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
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "resources/chalet.obj")) {
		throw std::runtime_error(warn + err);
	}

	std::unordered_map<Vertex, uint32_t> unique_vtx = {};

	for (const auto &shape : shapes) {
		for (const auto &index : shape.mesh.indices) {
			Vertex vertex = {};
			vertex.pos = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};
			vertex.tex_coord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};
			vertex.color = { 1.0f, 1.0f, 1.0f };
#if 1
			if (unique_vtx.count(vertex) == 0) {
				unique_vtx[vertex] = uint32_t(m_model_vertices.size());
				m_model_vertices.push_back(vertex);
			}

			m_model_indices.push_back(unique_vtx[vertex]);
#else 
			m_model_indices.push_back(m_model_vertices.size());
			m_model_vertices.push_back(vertex);
			
#endif
		}
		fprintf(stdout, "Loaded model: num vertices %lu, num indices %lu\n", 
				m_model_vertices.size(),
				m_model_indices.size());
	}
}

void BaseApplication::create_vertex_buffer()
{
	auto bufsize = sizeof(Vertex) * m_model_vertices.size();
	
	VkBuffer staging_buffer;
	VkDeviceMemory staging_memory;
	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				  staging_buffer, staging_memory);
	
	void *data;
	auto res = vkMapMemory(m_device, staging_memory, 0, bufsize, 0, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
	std::memcpy(data, m_model_vertices.data(), bufsize);
	vkUnmapMemory(m_device, staging_memory);

	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
				  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_vertex_buffer, m_vertex_buffer_memory);
	copy_buffer(staging_buffer, m_vertex_buffer, bufsize);

	vkDestroyBuffer(m_device, staging_buffer, nullptr);
	vkFreeMemory(m_device, staging_memory, nullptr);
}

void BaseApplication::create_index_buffer()
{
	
	VkDeviceSize bufsize = sizeof(uint32_t) * m_model_indices.size();
	VkBuffer staging_buf;
	VkDeviceMemory staging_mem;
	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				  staging_buf, staging_mem);

	void *data;
	auto res = vkMapMemory(m_device, staging_mem, 0, bufsize, 0, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map memory");
	std::memcpy(data, m_model_indices.data(), bufsize);
	vkUnmapMemory(m_device, staging_mem);

	create_buffer(bufsize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
				  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_index_buffer, m_index_buffer_memory);
	copy_buffer(staging_buf, m_index_buffer, bufsize);

	vkDestroyBuffer(m_device, staging_buf, nullptr);
	vkFreeMemory(m_device, staging_mem, nullptr);
}

void BaseApplication::create_uniform_buffers()
{
	VkDeviceSize bufsize = sizeof(CameraMatrices);
	m_uni_buffers.resize(m_swapchain_images.size());
	m_uni_buffer_memory.resize(m_swapchain_images.size());

	for (size_t i = 0; i < m_swapchain_images.size(); ++i) {
		create_buffer(bufsize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
					  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
					  m_uni_buffers[i], m_uni_buffer_memory[i]);
	}
}

void BaseApplication::create_bottom_acceleration_structure()
{
	auto indices = find_queue_families(m_gpu);
	uint32_t qidx[] = {
		indices.graphics_family.value(),
	};

	VkGeometryTrianglesNV trias = {};
	trias.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
	trias.indexCount = uint32_t(m_model_indices.size());
	trias.indexData = m_index_buffer;
	trias.indexOffset = 0;
	trias.indexType = VK_INDEX_TYPE_UINT32;
	trias.vertexCount = uint32_t(m_model_vertices.size());
	trias.vertexData = m_vertex_buffer;
	trias.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	trias.vertexOffset = 0;
	trias.vertexStride = sizeof(Vertex);
	trias.transformData = VK_NULL_HANDLE;

	VkGeometryNV g = {};
	g.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
	g.flags = VK_GEOMETRY_OPAQUE_BIT_NV;
	g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
	g.geometry.triangles = trias;
	g.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
	g.geometry.aabbs.numAABBs = 0;
	
	VkAccelerationStructureInfoNV si = {};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	si.instanceCount = 0;
	si.geometryCount = 1;
	si.pGeometries = &g;
	si.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;

	VkAccelerationStructureCreateInfoNV ci = {};
	ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	ci.info = si;

	auto res = evkCreateAccelerationStructureNV(m_device, &ci, nullptr, &m_bottom_as.structure);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create acceleration structure");
	
	// allocate scratch
	{
		VkMemoryRequirements2 mem_req = {};
		mem_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
		VkAccelerationStructureMemoryRequirementsInfoNV ri = {};
		ri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		ri.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		ri.accelerationStructure = m_bottom_as.structure;
		evkGetAccelerationStructureMemoryRequirementsNV(m_device, &ri, &mem_req);
	
		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = mem_req.memoryRequirements.size;
		ai.memoryTypeIndex = find_memory_type(mem_req.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		res = vkAllocateMemory(m_device, &ai, nullptr, &m_bottom_as.scratch_memory);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");
	
		VkBufferCreateInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size = mem_req.memoryRequirements.size;
		bi.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;
		bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bi.pQueueFamilyIndices = qidx;
		bi.queueFamilyIndexCount = 1;

		res = vkCreateBuffer(m_device, &bi, nullptr, &m_bottom_as.scratch_buffer);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to create buffer");

		res = vkBindBufferMemory(m_device, m_bottom_as.scratch_buffer, m_bottom_as.scratch_memory, 0);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to bind buffer memory");
		fprintf(stdout, "BOTTOM AS: needed scratch memory %lu MB\n", mem_req.memoryRequirements.size / 1024 / 1024);
	}

	// allocate scratch
	{
		VkMemoryRequirements2 mem_req = {};
		mem_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
		VkAccelerationStructureMemoryRequirementsInfoNV ri = {};
		ri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		ri.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		ri.accelerationStructure = m_bottom_as.structure;
		evkGetAccelerationStructureMemoryRequirementsNV(m_device, &ri, &mem_req);
		
		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = mem_req.memoryRequirements.size;
		ai.memoryTypeIndex = find_memory_type(mem_req.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		res = vkAllocateMemory(m_device, &ai, nullptr, &m_bottom_as.memory);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");

		VkBindAccelerationStructureMemoryInfoNV bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bi.accelerationStructure = m_bottom_as.structure;
		bi.memory = m_bottom_as.memory;
		bi.memoryOffset = 0;
		bi.deviceIndexCount = 0;

		res = evkBindAccelerationStructureMemoryNV(m_device, 1, &bi);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to bind acceleration structure memory");
		fprintf(stdout, "BOTTOM AS: needed structure memory %lu MB\n", mem_req.memoryRequirements.size / 1024 / 1024);
	}

	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);
	
	evkCmdBuildAccelerationStructureNV(cmd_buf, &si, VK_NULL_HANDLE, 0,
									  VK_FALSE, m_bottom_as.structure, VK_NULL_HANDLE,
									  m_bottom_as.scratch_buffer, 0);

	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::create_top_acceleration_structure()
{
	auto indices = find_queue_families(m_gpu);
	uint32_t qidx[] = {
		indices.graphics_family.value(),
	};

	VkAccelerationStructureInfoNV si = {};
	si.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	si.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
	si.instanceCount = 1;
	si.geometryCount = 0;

	VkAccelerationStructureCreateInfoNV ci = {};
	ci.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	ci.info = si;

	auto res = evkCreateAccelerationStructureNV(m_device, &ci, nullptr, &m_top_as.structure);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create acceleration structure");

	// allocate scratch
	{
		VkMemoryRequirements2 mem_req = {};
		mem_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
		VkAccelerationStructureMemoryRequirementsInfoNV ri = {};
		ri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		ri.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		ri.accelerationStructure = m_top_as.structure;
		evkGetAccelerationStructureMemoryRequirementsNV(m_device, &ri, &mem_req);

		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = mem_req.memoryRequirements.size;
		ai.memoryTypeIndex = find_memory_type(mem_req.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		res = vkAllocateMemory(m_device, &ai, nullptr, &m_top_as.scratch_memory);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");

		VkBufferCreateInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.size = mem_req.memoryRequirements.size;
		bi.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;
		bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bi.pQueueFamilyIndices = qidx;
		bi.queueFamilyIndexCount = 1;

		res = vkCreateBuffer(m_device, &bi, nullptr, &m_top_as.scratch_buffer);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to create buffer");

		res = vkBindBufferMemory(m_device, m_top_as.scratch_buffer, m_top_as.scratch_memory, 0);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to bind buffer memory");
		fprintf(stdout, "TOP AS: needed scratch memory %lu MB\n", mem_req.memoryRequirements.size / 1024 / 1024);
	}

	// allocate scratch
	{
		VkMemoryRequirements2 mem_req = {};
		mem_req.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
		VkAccelerationStructureMemoryRequirementsInfoNV ri = {};
		ri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		ri.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		ri.accelerationStructure = m_top_as.structure;
		evkGetAccelerationStructureMemoryRequirementsNV(m_device, &ri, &mem_req);

		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = mem_req.memoryRequirements.size;
		ai.memoryTypeIndex = find_memory_type(mem_req.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		res = vkAllocateMemory(m_device, &ai, nullptr, &m_top_as.memory);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");

		VkBindAccelerationStructureMemoryInfoNV bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bi.accelerationStructure = m_top_as.structure;
		bi.memory = m_top_as.memory;
		bi.memoryOffset = 0;
		bi.deviceIndexCount = 0;

		res = evkBindAccelerationStructureMemoryNV(m_device, 1, &bi);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to bind acceleration structure memory");
		fprintf(stdout, "TOP AS: needed structure memory %lu MB\n", mem_req.memoryRequirements.size / 1024 / 1024);
	}

	// configure instances
	{
		VkDeviceSize sz = sizeof(VkGeometryInstanceNV);
		VkBufferCreateInfo bi = {};
		bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bi.usage = VK_BUFFER_USAGE_RAY_TRACING_BIT_NV;
		bi.size = sz;
		bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bi.queueFamilyIndexCount = 1;
		bi.pQueueFamilyIndices = qidx;
		res = vkCreateBuffer(m_device, &bi, nullptr, &m_top_as.instances);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to create buffer");

		VkMemoryRequirements mem_req;
		vkGetBufferMemoryRequirements(m_device, m_top_as.instances, &mem_req);
		VkMemoryAllocateInfo ai = {};
		ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		ai.allocationSize = mem_req.size;
		ai.memoryTypeIndex = find_memory_type(mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		res = vkAllocateMemory(m_device, &ai, nullptr, &m_top_as.instances_memory);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate gpu memory");
		res = vkBindBufferMemory(m_device, m_top_as.instances, m_top_as.instances_memory, 0);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to bind buffer memory");

		uint64_t bottom_as_gpu_handle;
		res = evkGetAccelerationStructureHandleNV(m_device, m_bottom_as.structure, sizeof(uint64_t), &bottom_as_gpu_handle);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to get acceleration structure gpu handle");

		VkGeometryInstanceNV *instance_ptr;
		res = vkMapMemory(m_device, m_top_as.instances_memory, 0, sz, 0, (void**)&instance_ptr);
		if (res != VK_SUCCESS) throw std::runtime_error("failed to map buffer");
		glm::mat4 transform = glm::mat4(1.0f);
		transform = glm::transpose(transform);
		memcpy(instance_ptr->transform, &transform[0][0], sizeof(float)*12);
		instance_ptr->instanceCustomIndex = 0;
		instance_ptr->mask = 0xFF;
		instance_ptr->flags = 0;
		instance_ptr->instanceOffset = 0;
		instance_ptr->accelerationStructureHandle = bottom_as_gpu_handle;
		vkUnmapMemory(m_device, m_top_as.instances_memory);
	}

	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);

	evkCmdBuildAccelerationStructureNV(cmd_buf, &si, m_top_as.instances, 0,
									   VK_FALSE, m_top_as.structure, VK_NULL_HANDLE,
									   m_top_as.scratch_buffer, 0);

	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);

}

void BaseApplication::create_raytracing_pipeline_layout()
{
	VkDescriptorSetLayoutBinding lb_0 = {};
	lb_0.binding = 0;
	lb_0.descriptorCount = 1;
	lb_0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	lb_0.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;

	VkDescriptorSetLayoutBinding lb_1 = {};
	lb_1.binding = 1;
	lb_1.descriptorCount = 1;
	lb_1.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	lb_1.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;
	
	std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
		lb_0, lb_1
	};

	VkDescriptorSetLayoutCreateInfo li = {};
	li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	li.bindingCount = uint32_t(bindings.size());
	li.pBindings = bindings.data();
	auto res = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_rt_descriptor_set_layout);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create descriptor set layout");

	// Pipeline Layout
	VkPipelineLayoutCreateInfo plci = {};
	plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
	auto raygen_module = create_shader_module(read_file("resources/simple.rgen.spv"));
	auto chit_module = create_shader_module(read_file("resources/simple.rchit.spv"));
	auto miss_module = create_shader_module(read_file("resources/simple.rmiss.spv"));

	VkPipelineShaderStageCreateInfo rgci = {};
	rgci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	rgci.stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
	rgci.module = raygen_module;
	rgci.pName = "main";

	VkPipelineShaderStageCreateInfo chci = {};
	chci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	chci.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	chci.module = chit_module;
	chci.pName = "main";

	VkPipelineShaderStageCreateInfo mci = {};
	mci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	mci.stage = VK_SHADER_STAGE_MISS_BIT_NV;
	mci.module = miss_module;
	mci.pName = "main";

	std::array<VkPipelineShaderStageCreateInfo, 3> stages = { rgci, chci, mci };

	std::array<VkRayTracingShaderGroupCreateInfoNV, 3> groups = {};

	// raygen group
	groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	groups[0].generalShader = 0;
	groups[0].closestHitShader = VK_SHADER_UNUSED_NV;
	groups[0].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[0].intersectionShader = VK_SHADER_UNUSED_NV;
	// hitgroup
	groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
	groups[1].generalShader = VK_SHADER_UNUSED_NV;
	groups[1].closestHitShader = 1;
	groups[1].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[1].intersectionShader = VK_SHADER_UNUSED_NV;
	// miss group
	groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	groups[2].generalShader = 2;
	groups[2].closestHitShader = VK_SHADER_UNUSED_NV;
	groups[2].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[2].intersectionShader = VK_SHADER_UNUSED_NV;


	VkRayTracingPipelineCreateInfoNV ci = {};
	ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
	ci.flags = 0;
	ci.stageCount = uint32_t(stages.size());
	ci.pStages = stages.data();
	ci.groupCount = uint32_t(groups.size());
	ci.pGroups = groups.data();
	ci.maxRecursionDepth = 1;
	ci.layout = m_rt_pipeline_layout;
	

	auto res = evkCreateRayTracingPipelinesNV(m_device, VK_NULL_HANDLE, 1, &ci, nullptr, &m_rt_pipeline);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create a raytracing pipeline");


	vkDestroyShaderModule(m_device, raygen_module, nullptr);
	vkDestroyShaderModule(m_device, chit_module, nullptr);
	vkDestroyShaderModule(m_device, miss_module, nullptr);
}

void BaseApplication::create_texture_image()
{
	int tex_width, tex_height, tex_channels;
	stbi_uc *pixels = stbi_load("resources/chalet.jpg", 
			&tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
	if (!pixels) {
		throw std::runtime_error("failed to load texture");
	}
	VkDeviceSize image_size = tex_width * tex_height * 4;

	VkBuffer staging_buffer;
	VkDeviceMemory staging_memory;
	create_buffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				  staging_buffer, staging_memory);

	void *data;
	auto res = vkMapMemory(m_device, staging_memory, 0, image_size, 0, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map buffer memory");
	std::memcpy(data, pixels, size_t(image_size));
	vkUnmapMemory(m_device, staging_memory);

	stbi_image_free(pixels);

	create_image(tex_width, tex_height, VK_FORMAT_R8G8B8A8_UNORM,
				 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_texture_img, m_texture_img_memory);

	transition_image_layout(m_texture_img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED,
							VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
	copy_buffer_to_image(staging_buffer, m_texture_img, tex_width, tex_height);
	transition_image_layout(m_texture_img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
							VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


	vkDestroyBuffer(m_device, staging_buffer, nullptr);
	vkFreeMemory(m_device, staging_memory, nullptr);
}

void BaseApplication::create_texture_image_view()
{
	m_texture_img_view = vk_helpers::create_image_view_2d(m_device, m_texture_img, 
			VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}

void BaseApplication::create_texture_sampler()
{
	VkSamplerCreateInfo si = {};
	si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	si.magFilter = VK_FILTER_LINEAR;
	si.minFilter = VK_FILTER_LINEAR;
	si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	si.anisotropyEnable = VK_TRUE;
	si.maxAnisotropy = 16;
	si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	si.unnormalizedCoordinates = VK_FALSE;
	si.compareEnable = VK_FALSE;
	si.compareOp = VK_COMPARE_OP_ALWAYS;
	si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	si.mipLodBias = 0.0f;
	si.minLod = 0.0f;
	si.maxLod = 0.0f;

	auto res = vkCreateSampler(m_device, &si, nullptr, &m_texture_sampler);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create texture sampler");
}

void BaseApplication::create_rt_image()
{
	create_image(m_swapchain_extent.width, m_swapchain_extent.height, VK_FORMAT_R8G8B8A8_UNORM,
				 VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_rt_img, m_rt_img_memory);

	m_rt_img_view = vk_helpers::create_image_view_2d(m_device, m_rt_img, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

	// we do the transition with a pipeline barrier as it is needed to be done only once
	auto cmd_buf = begin_single_time_commands(m_graphics_queue, m_graphics_cmd_pool);

	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = m_rt_img;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
	VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV;
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

	vkCmdPipelineBarrier(cmd_buf, src_stage, dst_stage,
						 0, 0, nullptr, 0, nullptr, 1, &barrier);

	end_single_time_commands(m_graphics_queue, m_graphics_cmd_pool, cmd_buf);
}

void BaseApplication::create_descriptor_pool()
{
	uint32_t imgs_count = (uint32_t)m_swapchain_images.size();
	std::array<VkDescriptorPoolSize, 4> ps = {};
	ps[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	ps[0].descriptorCount = imgs_count;
	ps[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	ps[1].descriptorCount = imgs_count;
	// rt descriptors 
	ps[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	ps[2].descriptorCount = imgs_count;
	ps[3].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	ps[3].descriptorCount = imgs_count;

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
		bi.buffer = m_uni_buffers[i];
		bi.offset = 0;
		bi.range = sizeof(CameraMatrices);
		VkDescriptorImageInfo ii = {};
		ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		ii.imageView = m_texture_img_view;
		ii.sampler = m_texture_sampler;

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
		dw[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		dw[1].descriptorCount = 1;
		dw[1].pImageInfo = &ii;
		
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
		
		VkWriteDescriptorSetAccelerationStructureNV dw_rt = {};
		dw_rt.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
		dw_rt.accelerationStructureCount = 1;
		dw_rt.pAccelerationStructures = &m_top_as.structure;
		
		std::array<VkWriteDescriptorSet, 2> dw = {};
		dw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[0].dstSet = m_rt_desc_sets[i];
		dw[0].dstBinding = 0;
		dw[0].dstArrayElement = 0;
		dw[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		dw[0].descriptorCount = 1;
		dw[0].pImageInfo = &ii;

		dw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		dw[1].dstSet = m_rt_desc_sets[i];
		dw[1].dstBinding = 1;
		dw[1].dstArrayElement = 0;
		dw[1].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
		dw[1].descriptorCount = 1;
		dw[1].pNext = &dw_rt;

		vkUpdateDescriptorSets(m_device, uint32_t(dw.size()), dw.data(), 0, nullptr);
	}

}

void BaseApplication::create_shader_binding_table()
{
	VkPhysicalDeviceRayTracingPropertiesNV props = vk_helpers::get_raytracing_properties(m_gpu);
	
	//vkGetRayTracingShaderGroupHandlesNV(m_device, m_rt_pipeline, )
	//vkGetRayTracingShaderGroupHandlesNV()
	//props.shaderGroupHandleSize 
	//vkGetRayTracingShaderGroupHandlesNV()
	//create_buffer()
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
				 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depth_img, m_depth_img_memory);

	m_depth_img_view = vk_helpers::create_image_view_2d(m_device, m_depth_img, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);

	// we could do the layout transition in the renderpass as the color attachment but here 
	// we do it with a pipeline barrier as it is needed to be done only once
	transition_image_layout(m_depth_img, depth_format,
							VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

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
		bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
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
		
		VkBuffer buffers[] = { m_vertex_buffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(m_cmd_buffers[i], 0, 1, buffers, offsets);
		vkCmdBindIndexBuffer(m_cmd_buffers[i], m_index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout,
								0, 1, &m_desc_sets[i], 0, nullptr);
		vkCmdDrawIndexed(m_cmd_buffers[i], uint32_t(m_model_indices.size()), 1, 0 , 0, 0);

		vkCmdEndRenderPass(m_cmd_buffers[i]);

		res = vkEndCommandBuffer(m_cmd_buffers[i]);
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
	auto curr_time = std::chrono::high_resolution_clock::now();
	float time = std::chrono::duration<float, std::chrono::seconds::period>(curr_time - start_time).count();

	CameraMatrices ubo = {};
	ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(10.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	ubo.proj = glm::perspective(glm::radians(45.0f), m_swapchain_extent.width / (float)m_swapchain_extent.height, 0.1f, 10.0f);
	ubo.proj[1][1] *= -1;

	void *data;
	auto res = vkMapMemory(m_device, m_uni_buffer_memory[idx], 0, sizeof(CameraMatrices), 0, &data);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to map uniform buffer memory");
	std::memcpy(data, &ubo, sizeof(CameraMatrices));
	vkUnmapMemory(m_device, m_uni_buffer_memory[idx]);
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

	VkSubmitInfo si = {};
	si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	VkSemaphore wait_semaphores[] = { m_sem_img_available[m_current_frame_idx] };
	VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	si.waitSemaphoreCount = 1;
	si.pWaitSemaphores = wait_semaphores;
	si.pWaitDstStageMask = wait_stages;
	si.commandBufferCount = 1;
	si.pCommandBuffers = &m_cmd_buffers[img_idx];
	VkSemaphore signal_semaphores[] = { m_sem_render_finished[m_current_frame_idx] };
	si.signalSemaphoreCount = 1;
	si.pSignalSemaphores = signal_semaphores;

	// we reset fences here because we need it after checking for swapchain recreation
	// else we could apply it after vkWaitForFences
	vkResetFences(m_device, 1, &m_fen_flight[m_current_frame_idx]);
	res = vkQueueSubmit(m_graphics_queue, 1, &si, m_fen_flight[m_current_frame_idx]);
	if (res != VK_SUCCESS) {
		throw std::runtime_error("failed to submit command buffers to queue");
	}

	VkPresentInfoKHR pi = {};
	pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	pi.waitSemaphoreCount = 1;
	pi.pWaitSemaphores = signal_semaphores;
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