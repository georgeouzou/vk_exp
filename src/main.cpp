#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <functional>
#include <vector>

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

private:
	GLFWwindow *m_window{ nullptr };
	uint32_t m_width{ 1024 };
	uint32_t m_height{ 768 };

	VkInstance m_instance{ VK_NULL_HANDLE };
	VkDebugUtilsMessengerEXT m_debug_callback{ VK_NULL_HANDLE };

	std::vector<const char*> m_validation_layers;
	bool m_enable_validation_layers;
};

void BaseApplication::run()
{
	init_window();
	init_vulkan();
	main_loop();
	cleanup();
}

BaseApplication::BaseApplication()
{
	m_validation_layers.push_back("VK_LAYER_LUNARG_standard_validation");
#if defined(NDEBUG)
	m_enable_validation_layers = false;
#else 
	m_enable_validation_layers = true;
#endif
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
}

void BaseApplication::main_loop()
{
	while (!glfwWindowShouldClose(m_window)) {
		glfwPollEvents();
	}
}

void BaseApplication::cleanup()
{
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