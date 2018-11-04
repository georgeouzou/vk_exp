cmake_minimum_required(VERSION 3.7)

set(app VulkanExperiments)

project(${app})

set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")

set(src src/main.cpp)

add_executable(${app} ${src})
set_property(TARGET ${app} PROPERTY CXX_STANDARD 11)

# glfw

set(GLFW_DIR "${EXTERN_DIR}/glfw")
set(GLFW_BUILD_DOCS     OFF CACHE INTERNAL "Build the GLFW documentation")
set(GLFW_BUILD_TESTS    OFF CACHE INTERNAL "Build the GLFW tests")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW examples")
set(GLFW_INSTALL        OFF CACHE INTERNAL "Generate installation target")
add_subdirectory(${GLFW_DIR})
target_link_libraries(${app} glfw ${GLFW_LIBRARIES})
target_include_directories(${app} PRIVATE ${GLFW_DIR}/include)
target_compile_definitions(${app} PRIVATE "GLFW_INCLUDE_NONE")

# vulkan

find_package(Vulkan)
if (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find vulkan library!")
endif()
target_link_libraries(${app} ${Vulkan_LIBRARIES})
target_include_directories(${app} PRIVATE ${Vulkan_INCLUDE_DIRS})

if (WIN32)
	target_compile_definitions(${app} PRIVATE "VK_USE_PLATFORM_WIN32_KHR")
	target_compile_definitions(${app} PRIVATE "GLFW_EXPOSE_NATIVE_WIN32")
endif()