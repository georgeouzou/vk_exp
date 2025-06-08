#ifndef CUDA_SHARED_HPP
#define CUDA_SHARED_HPP

#include <array>
#include <volk.h>
#include <cuda_runtime.h>

class CudaSharedBuffer
{
public:
	CudaSharedBuffer(VkPhysicalDevice gpu, VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage);
	~CudaSharedBuffer();

	inline VkBuffer get_vk_buffer() const { return m_buffer; }

	template<typename T>
	inline T* get_cuda_buffer() const { return reinterpret_cast<T*>(m_cuda_buffer); };
	inline VkDeviceSize size() const { return m_size; }
private:
	VkDevice m_device{ VK_NULL_HANDLE };
	VkBuffer m_buffer{ VK_NULL_HANDLE };
	VkDeviceMemory m_memory{ VK_NULL_HANDLE };
	cudaExternalMemory_t m_cuda_memory{ nullptr };
	uint8_t* m_cuda_buffer{ nullptr };
	VkDeviceSize m_size{ 0 };
};

class CudaSharedSemaphore
{
public:
	CudaSharedSemaphore(VkDevice device);
	~CudaSharedSemaphore();

	void increment();
	uint64_t get_value();

	inline VkSemaphore get_vk_semaphore() const { return m_vk_timeline_sem; }
	inline cudaExternalSemaphore_t get_cuda_semaphore() const { return m_cuda_timeline_sem; }
private:
	VkDevice m_device{ VK_NULL_HANDLE };
	VkSemaphore m_vk_timeline_sem{ VK_NULL_HANDLE };
	cudaExternalSemaphore_t m_cuda_timeline_sem{ nullptr };
	uint64_t m_timeline_value{ 0 };
};

class CudaDenoiser
{
public:
	CudaDenoiser(std::array<uint8_t, VK_UUID_SIZE> uuid);
	~CudaDenoiser();

	void denoise(CudaSharedBuffer &dst, const CudaSharedBuffer &src, CudaSharedSemaphore& sem);
private:
	int select_cuda_device(std::array<uint8_t, VK_UUID_SIZE> uuid);

private:
	cudaStream_t m_stream;
};

#endif