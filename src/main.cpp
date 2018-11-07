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

VkPhysicalDevice get_physical_device(VkInstance instance)
{
	std::vector<VkPhysicalDevice> phys_devices(2);
	uint32_t phys_device_count = (uint32_t)phys_devices.size();
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &phys_device_count, phys_devices.data()));

	VkPhysicalDevice phys_device = choose_physical_device(phys_devices);
	return phys_device;
}

VkDevice create_device(VkInstance instance, VkPhysicalDevice phys_device, uint32_t *family_index)
{
	uint32_t queue_family_count;
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_family_count, queue_family_properties.data());

	*family_index = 0;

	float queue_priorities[] = { 1.0f };
	VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qci.queueFamilyIndex = 0;
	qci.queueCount = 1;
	qci.pQueuePriorities = queue_priorities;

	const char *extensions[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};

	VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	dci.pQueueCreateInfos = &qci;
	dci.queueCreateInfoCount = 1;
	dci.enabledExtensionCount = sizeof(extensions) / sizeof(extensions[0]);
	dci.ppEnabledExtensionNames = extensions; 
	
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

VkFormat get_supported_format(VkPhysicalDevice phys_device, VkSurfaceKHR surface)
{
	uint32_t fmt_count;
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &fmt_count, nullptr));
	std::vector<VkSurfaceFormatKHR> fmts(fmt_count);
	VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(phys_device, surface, &fmt_count, fmts.data()));
	return fmts[0].format;
}

VkSwapchainKHR create_swapchain(VkPhysicalDevice phys_device, VkDevice device, VkSurfaceKHR surface, 
								uint32_t family_idx, uint32_t width, uint32_t height, VkFormat fmt)
{
	// check if surface is supported for presentation
	VkBool32 supported;
	VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(phys_device, family_idx, surface, &supported));
	if (!supported) return VK_NULL_HANDLE;

	VkSwapchainKHR swapchain = 0;
	VkSwapchainCreateInfoKHR scci = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
	scci.surface = surface;
	scci.minImageCount = 2;
	scci.imageFormat = fmt;
	scci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
	scci.imageExtent = VkExtent2D{ width, height };
	scci.imageArrayLayers = 1;
	scci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
	scci.queueFamilyIndexCount = 1;
	scci.pQueueFamilyIndices = &family_idx;
	scci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
	scci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	scci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
	VK_CHECK(vkCreateSwapchainKHR(device, &scci, 0, &swapchain));
	return swapchain;
}

VkSemaphore create_semaphore(VkDevice device)
{
	VkSemaphore semaphore = 0;
	VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &semaphore));
	return semaphore;
}

VkCommandPool create_cmd_pool(VkDevice device, uint32_t family_idx)
{
	VkCommandPool cmd_pool = 0;
	VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	cpci.queueFamilyIndex = family_idx;
	cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool));
	return cmd_pool;
}

VkRenderPass create_render_pass(VkDevice device, VkFormat fmt)
{
	VkAttachmentDescription attachments[1] = {};
	attachments[0].format = fmt;
	attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_attachments_ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

	VkSubpassDescription subpass = { };
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &color_attachments_ref;

	VkRenderPassCreateInfo rpci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
	rpci.attachmentCount = sizeof(attachments) / sizeof(attachments[0]);
	rpci.pAttachments = attachments;
	rpci.subpassCount = 1;
	rpci.pSubpasses = &subpass;

	VkRenderPass render_pass = 0;
	VK_CHECK(vkCreateRenderPass(device, &rpci, 0, &render_pass));
	return render_pass;
}

VkFramebuffer create_framebuffer(VkDevice device, VkRenderPass render_pass, 
								 VkImageView image_view, uint32_t width, uint32_t height)
{
	VkFramebufferCreateInfo fbci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
	fbci.renderPass = render_pass;
	fbci.attachmentCount = 1;
	fbci.pAttachments = &image_view;
	fbci.width = width;
	fbci.height = height;
	fbci.layers = 1;

	VkFramebuffer fb = 0;
	VK_CHECK(vkCreateFramebuffer(device, &fbci, nullptr, &fb));
	return fb;
}

VkImageView create_image_view(VkDevice device, VkImage image, VkFormat fmt)
{
	VkImageViewCreateInfo ivc = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
	ivc.image = image;
	ivc.viewType = VK_IMAGE_VIEW_TYPE_2D;
	ivc.format = fmt;
	ivc.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	ivc.subresourceRange.layerCount = 1;
	ivc.subresourceRange.levelCount = 1;
	
	VkImageView img_view = 0;
	VK_CHECK(vkCreateImageView(device, &ivc, nullptr, &img_view));
	return img_view;
}

int main()
{
	int rc = glfwInit();
	assert(rc);
		
	VkInstance instance = create_instance();
	assert(instance);

	VkPhysicalDevice phys_device = get_physical_device(instance);
	assert(phys_device);

	uint32_t family_idx;
	VkDevice device = create_device(instance, phys_device, &family_idx);
	assert(device);

	GLFWwindow *window = glfwCreateWindow(1024, 720, "vk_exp", nullptr, nullptr);
	assert(window);

	VkSurfaceKHR surface = create_surface(instance, window);
	assert(surface);

	VkFormat format = get_supported_format(phys_device, surface);
	
	int w, h;
	glfwGetWindowSize(window, &w, &h);
	VkSwapchainKHR swapchain = create_swapchain(phys_device, device, surface, family_idx, w, h, format);
	assert(swapchain);

	VkSemaphore aquire_semaphore = create_semaphore(device);
	assert(aquire_semaphore);
	VkSemaphore submit_semaphore = create_semaphore(device);
	assert(submit_semaphore);

	VkQueue queue;
	vkGetDeviceQueue(device, family_idx, 0, &queue);

	uint32_t swapchain_image_count;
	vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr);
	std::vector<VkImage> swapchain_images(swapchain_image_count);
	vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data());

	VkRenderPass render_pass = create_render_pass(device, format);
	assert(render_pass);

	std::vector<VkImageView> image_views(swapchain_image_count);
	for (uint32_t i = 0; i < swapchain_image_count; ++i) {
		image_views[i] = create_image_view(device, swapchain_images[i], format);
	}

	std::vector<VkFramebuffer> framebuffers(swapchain_image_count);
	for (uint32_t i = 0; i < swapchain_image_count; ++i) {
		framebuffers[i] = create_framebuffer(device, render_pass, image_views[i], w, h);
	}

	VkCommandPool cmd_pool = create_cmd_pool(device, family_idx);
	assert(cmd_pool);

	VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cbi.commandPool = cmd_pool;
	cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbi.commandBufferCount = 1;

	VkCommandBuffer cmd_buf;
	VK_CHECK(vkAllocateCommandBuffers(device, &cbi, &cmd_buf));

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		uint32_t image_idx = 0;
		VK_CHECK(vkAcquireNextImageKHR(device, swapchain, ~0, aquire_semaphore, VK_NULL_HANDLE, &image_idx));
		VK_CHECK(vkResetCommandPool(device, cmd_pool, 0));

		VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK(vkBeginCommandBuffer(cmd_buf, &cbbi));
		
		VkClearColorValue color = { 0.3f, 0.6f, 0.9f, 1.0f };
		VkClearValue clear_value;
		clear_value.color = color;

		VkRenderPassBeginInfo rp_begin_info = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rp_begin_info.renderPass = render_pass;
		rp_begin_info.framebuffer = framebuffers[image_idx];
		rp_begin_info.renderArea.extent.width = w;
		rp_begin_info.renderArea.extent.height = h;
		rp_begin_info.clearValueCount = 1;
		rp_begin_info.pClearValues = &clear_value;
	
		vkCmdBeginRenderPass(cmd_buf, &rp_begin_info, VK_SUBPASS_CONTENTS_INLINE);
		
		vkCmdEndRenderPass(cmd_buf);

		VK_CHECK(vkEndCommandBuffer(cmd_buf));
		
		VkPipelineStageFlags submit_stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
		si.commandBufferCount = 1;
		si.pCommandBuffers = &cmd_buf;
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &aquire_semaphore;
		si.pWaitDstStageMask = &submit_stage_mask;
		si.signalSemaphoreCount = 1;
		si.pSignalSemaphores = &submit_semaphore;

		VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
	
		VkPresentInfoKHR pi = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
		pi.swapchainCount = 1;
		pi.pSwapchains = &swapchain;
		pi.pImageIndices = &image_idx;
		pi.waitSemaphoreCount = 1;
		pi.pWaitSemaphores = &submit_semaphore;

		VK_CHECK(vkQueuePresentKHR(queue, &pi));

		VK_CHECK(vkDeviceWaitIdle(device));
	}

	vkDestroyDevice(device, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);
	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}
