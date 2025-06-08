#include "cuda_shared.hpp"
#include <stdexcept>
#include <cassert>
#include <Windows.h>

namespace 
{

uint32_t find_memory_type(VkPhysicalDevice gpu, uint32_t type_filter, VkMemoryPropertyFlags props)
{
	VkPhysicalDeviceMemoryProperties mem_props;
	vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
		if (type_filter & (1 << i) &&
			(mem_props.memoryTypes[i].propertyFlags & props) == props) {
			return i;
		}
	}

	throw std::runtime_error("failed to find suitable memory type");
	return 0;
}

}

CudaSharedBuffer::CudaSharedBuffer(VkPhysicalDevice gpu, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage):
	m_device(device),
	m_size(size)
{
	VkExternalMemoryBufferCreateInfo embc = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, 0 };
	embc.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0 };
	bci.size = size;
	bci.usage = usage;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bci.pNext = &embc;

	auto res = vkCreateBuffer(m_device, &bci, nullptr, &m_buffer);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create buffer");
	VkMemoryRequirements mem_req;
	vkGetBufferMemoryRequirements(m_device, m_buffer, &mem_req);

	VkMemoryDedicatedAllocateInfo dai = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, 0 };
	dai.buffer = m_buffer;
	VkExportMemoryAllocateInfo emai = { VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, 0 };
	emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	emai.pNext = &dai;
	VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0 };
	mai.allocationSize = mem_req.size;
	mai.memoryTypeIndex = find_memory_type(gpu, mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	mai.pNext = &emai;
	res = vkAllocateMemory(m_device, &mai, nullptr, &m_memory);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to allocate memory");

	res = vkBindBufferMemory(m_device, m_buffer, m_memory, 0);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to bind buffer memory");

	VkMemoryGetWin32HandleInfoKHR hinfo = { VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR, 0 };
	hinfo.memory = m_memory;
	hinfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	HANDLE mem_handle;
	res = vkGetMemoryWin32HandleKHR(m_device, &hinfo, &mem_handle);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to share win32 memory");

	cudaExternalMemoryHandleDesc mem_handle_desc = {};
	mem_handle_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	mem_handle_desc.size = size;
	mem_handle_desc.handle.win32.handle = mem_handle;
	cudaImportExternalMemory(&m_cuda_memory, &mem_handle_desc);
	CloseHandle(mem_handle);

	cudaExternalMemoryBufferDesc buffer_desc = {};
	buffer_desc.offset = 0;
	buffer_desc.size = size;

	cudaExternalMemoryGetMappedBuffer((void**)&m_cuda_buffer, m_cuda_memory, &buffer_desc);
	if (m_cuda_buffer == nullptr) {
		throw std::runtime_error("could not create CUDA buffer");
	}
}

CudaSharedBuffer::~CudaSharedBuffer()
{
	cudaFree(m_cuda_buffer);
	cudaDestroyExternalMemory(m_cuda_memory);
	vkDestroyBuffer(m_device, m_buffer, nullptr);
	vkFreeMemory(m_device, m_memory, nullptr);
}


CudaSharedSemaphore::CudaSharedSemaphore(VkDevice device):
	m_device(device)
{
	VkSemaphoreTypeCreateInfo stci = { VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, 0 };
	stci.initialValue = m_timeline_value;
	stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
	VkExportSemaphoreCreateInfoKHR esci = { VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR, 0 };
	esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;
	esci.pNext = &stci;
	VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, 0 };
	sci.pNext = &esci;
	auto res = vkCreateSemaphore(m_device, &sci, nullptr, &m_vk_timeline_sem);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create timeline semaphore");

	VkSemaphoreGetWin32HandleInfoKHR hi = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR, 0 };
	hi.semaphore = m_vk_timeline_sem;
	hi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
	HANDLE h;
	res = vkGetSemaphoreWin32HandleKHR(m_device, &hi, &h);
	if (res != VK_SUCCESS) throw std::runtime_error("failed to create timeline semaphore win32  handle");

	cudaExternalSemaphoreHandleDesc sem_handle_desc = {};
	sem_handle_desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
	sem_handle_desc.handle.win32.handle = h;
	cudaImportExternalSemaphore(&m_cuda_timeline_sem, &sem_handle_desc);

	CloseHandle(h);
}

CudaSharedSemaphore::~CudaSharedSemaphore()
{
	vkDestroySemaphore(m_device, m_vk_timeline_sem, nullptr);
	cudaDestroyExternalSemaphore(m_cuda_timeline_sem);
}

void CudaSharedSemaphore::increment()
{
	m_timeline_value++;
}

uint64_t CudaSharedSemaphore::get_value()
{
	return m_timeline_value;
}

int CudaDenoiser::select_cuda_device(std::array<uint8_t, VK_UUID_SIZE> uuid)
{
	int current_device = 0;
	int device_count = 0;
	int devices_prohibited = 0;

	cudaGetDeviceCount(&device_count);

	while (current_device < device_count) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, current_device);

		if (props.computeMode != cudaComputeModeProhibited) {
			int ret = memcmp(uuid.data(), props.uuid.bytes, VK_UUID_SIZE);
			if (ret == 0) {
				fprintf(stdout, "CUDA selected device %s\n", props.name);
				return current_device;
			}
		} else {
			devices_prohibited++;
		}
		current_device++;
	}

	throw std::runtime_error("no Vulkan-CUDA interop");
	return -1;
}

CudaDenoiser::CudaDenoiser(std::array<uint8_t, VK_UUID_SIZE> uuid)
{
	int device_id = select_cuda_device(uuid);
	cudaSetDevice(device_id);
	cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
}

CudaDenoiser::~CudaDenoiser()
{
	cudaStreamDestroy(m_stream);
}

void CudaDenoiser::denoise(CudaSharedBuffer &dst, const CudaSharedBuffer &src, CudaSharedSemaphore& sem)
{
	{
		cudaExternalSemaphoreWaitParams p = {};
		p.params.fence.value = sem.get_value();
		auto cu_sem = sem.get_cuda_semaphore();
		cudaWaitExternalSemaphoresAsync(&cu_sem, &p, 1, m_stream);
	}

	// do nothing for now, just copy the src buffer
	assert(src.size() == dst.size());
	cudaMemcpyAsync(dst.get_cuda_buffer<uint8_t*>(), src.get_cuda_buffer<uint8_t*>(), src.size(), cudaMemcpyDeviceToDevice, m_stream);

	{
		sem.increment();
		cudaExternalSemaphoreSignalParams p = {};
		p.params.fence.value = sem.get_value();
		auto cu_sem = sem.get_cuda_semaphore();
		cudaSignalExternalSemaphoresAsync(&cu_sem, &p, 1, m_stream);
	}
}
