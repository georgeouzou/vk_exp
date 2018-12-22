#define GLFW_INCLUDE_VULKAN
#define NOMINMAX
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include <vector>
#include <optional>
#include <set>
#include <limits>
#include <algorithm>

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) { 
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
	VkDebugUtilsMessageTypeFlagsEXT message_type,
	const VkDebugUtilsMessengerCallbackDataEXT *p_callback_data,
	void *p_user_data)
{
	fprintf(stderr, "validation layer: %s\n", p_callback_data->pMessage);
	return VK_FALSE;
}

struct QueueFamilyIndices
{
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;
	
	bool is_complete()
	{
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct SwapchainSupportDetails
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

class BaseApplication
{
public:
	BaseApplication();
	~BaseApplication();
	void run();

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


private:
	GLFWwindow *m_window{ nullptr };
	uint32_t m_width{ 1024 };
	uint32_t m_height{ 768 };

	std::vector<const char*> m_validation_layers;
	bool m_enable_validation_layers;

	std::vector<const char*> m_device_extensions;

	VkInstance m_instance{ VK_NULL_HANDLE };
	VkDebugUtilsMessengerEXT m_debug_callback{ VK_NULL_HANDLE };
	VkPhysicalDevice m_gpu{ VK_NULL_HANDLE };

	VkDevice m_device{ VK_NULL_HANDLE };
	VkQueue m_graphics_queue{ VK_NULL_HANDLE };
	VkQueue m_present_queue{ VK_NULL_HANDLE };
	
	VkSurfaceKHR m_surface{ VK_NULL_HANDLE };

	VkSwapchainKHR m_swapchain{ VK_NULL_HANDLE };
};

void BaseApplication::run()
{
	init_window();
	init_vulkan();
	main_loop();
}

BaseApplication::BaseApplication()
{
	m_validation_layers.push_back("VK_LAYER_LUNARG_standard_validation");
#if defined(NDEBUG)
	m_enable_validation_layers = false;
#else 
	m_enable_validation_layers = true;
#endif

	m_device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
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
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	m_window = glfwCreateWindow(m_width, m_height, "tutorial", NULL, NULL);
	if (!m_window) {
		throw std::runtime_error("could not create glfw window");
	}

	glfwSetKeyCallback(m_window, key_callback);
}

void BaseApplication::init_vulkan()
{
	create_instance();
	if (m_enable_validation_layers) {
		setup_debug_callback();
	}

	create_surface();

	pick_gpu();
	create_logical_device();

	create_swapchain();
}

void BaseApplication::main_loop()
{
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
	}
}

void BaseApplication::cleanup()
{
	if (m_swapchain) vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

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
	ai.apiVersion = VK_API_VERSION_1_0;

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
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(gpu, &props);
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
	
	return indices.is_complete() && extensions_supported && swapchain_adequate;
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
		VkExtent2D actual = { m_width, m_height };
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