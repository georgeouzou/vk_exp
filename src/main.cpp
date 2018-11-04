#include <iostream>
#include <cassert>
#include <vector>
#include <cstdio>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(call)\
	do {\
		VkResult __res = call;\
		assert(__res == VK_SUCCESS);\
	} while(0)

VkPhysicalDevice choose_physical_device(const std::vector<VkPhysicalDevice> &devices)
{
	for (auto d : devices) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(d, &props);
		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			fprintf(stdout, "Picking discrete gpu %s\n", props.deviceName);
			return d;
		}
	}
	fprintf(stdout, "Could not pick a physical device\n");
	return VK_NULL_HANDLE;
}

VkInstance create_instance()
{
	VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	ai.apiVersion = VK_API_VERSION_1_1;

	VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	ici.pApplicationInfo = &ai;
#if defined(_DEBUG)
	const char *layers[] = {
		"VK_LAYER_LUNARG_standard_validation"
	};
	ici.ppEnabledLayerNames = layers;
	ici.enabledLayerCount = sizeof(layers) / sizeof(layers[0]);
#endif

#if defined(VK_USE_PLATFORM_WIN32_KHR)
	const char *extensions[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
		VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
	};
	ici.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
	ici.ppEnabledExtensionNames = extensions;
#endif
	
	VkInstance instance = VK_NULL_HANDLE;
	VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
	return instance;
}

VkDevice create_device(VkInstance instance)
{
	std::vector<VkPhysicalDevice> phys_devices(2);
	uint32_t phys_device_count = phys_devices.size();
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &phys_device_count, phys_devices.data()));

	VkPhysicalDevice phys_device = choose_physical_device(phys_devices);
	assert(phys_device);

	uint32_t queue_family_count;
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, queue_family_properties.data());

	float queue_priorities[] = { 1.0f };
	VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qci.queueFamilyIndex = 0;
	qci.queueCount = 1;
	qci.pQueuePriorities = queue_priorities;

	VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	dci.pQueueCreateInfos = &qci;
	dci.queueCreateInfoCount = 1;
	
	VkDevice device = VK_NULL_HANDLE;
	VK_CHECK(vkCreateDevice(phys_device, &dci, 0, &device));
	return device;
}

VkSurfaceKHR create_surface(VkInstance instance, GLFWwindow *window)
{
	VkSurfaceKHR surface = VK_NULL_HANDLE;
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	VkWin32SurfaceCreateInfoKHR sci = { VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
	sci.hinstance = GetModuleHandle(0);
	sci.hwnd = glfwGetWin32Window(window);
	VK_CHECK(vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface));
#else 
#error "unsupported platform"
#endif
	return surface;
}

int main()
{
	int rc = glfwInit();
	assert(rc);
		
	VkInstance instance = create_instance();
	assert(instance);

	VkDevice device = create_device(instance);
	assert(device);

	GLFWwindow *window = glfwCreateWindow(1024, 720, "vk_exp", nullptr, nullptr);
	assert(window);

	VkSurfaceKHR surface = create_surface(instance, window);
	assert(surface);

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
	}

	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
