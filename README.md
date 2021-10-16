# vk_exp

Experimenting with vulkan basics and ray tracing features

Based on
- https://vulkan-tutorial.com/ 
- the vulkan spec
- material shading of [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

To run you need to have a gpu that supports vulkan KHR raytracing

## How to init and build

* git clone https://github.com/georgeouzou/vk_exp.git
* cd vk_exp
* git submodule update --init --recursive
* Linux: mkdir build; cd build; cmake ..; make
* Windows: open as cmake local folder in Visual Studio 2019 

## Licenses and Open Source Software

The code uses the following dependencies:
* Latest vulkan sdk (supporting VK_KHR_raytracing_pipeline)
* [TinyObjLoader](https://github.com/syoyo/tinyobjloader-c/blob/master/README.md)
* [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)
* [VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
* [glfw](https://github.com/glfw/glfw)
* [glm](https://github.com/g-truc/glm)
* [volk](https://github.com/zeux/volk)

Models downloaded from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)
