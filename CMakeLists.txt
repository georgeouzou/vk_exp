cmake_minimum_required(VERSION 3.7)

set(app VulkanExperiments)

project(${app})

set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options (-fcolor-diagnostics)
endif()

set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/shader_dir.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/src/shader_dir.h" @ONLY)

set(src 
	src/main.cpp 
	src/stb_image.c
	src/tiny_obj_loader.cpp
	src/orbit_camera.cpp
	src/vma.cpp
)

add_executable(${app} ${src})
set_property(TARGET ${app} PROPERTY CXX_STANDARD 17)

if (WIN32)
	add_definitions(-DVK_USE_PLATFORM_WIN32_KHR)
	add_definitions(-DNOMINMAX)
endif()

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

# glm
set(GLM_DIR "${EXTERN_DIR}/glm")
target_include_directories(${app} PRIVATE ${GLM_DIR})

# tinyobjloader
set(TINYOBJLOADER "${EXTERN_DIR}/tinyobjloader")
target_include_directories(${app} PRIVATE ${TINYOBJLOADER})

# volk
set(VOLK "${EXTERN_DIR}/volk")
add_subdirectory(${VOLK})
target_include_directories(${app} PRIVATE ${VOLK})
target_link_libraries(${app} volk)

# vulkan
find_package(Vulkan)
if (NOT Vulkan_FOUND)
	message(FATAL_ERROR "Could not find vulkan library!")
endif()

target_include_directories(${app} PRIVATE ${Vulkan_INCLUDE_DIRS})
if (WIN32)
	target_link_libraries(${app} $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
else()
	target_link_directories(${app} PRIVATE $ENV{VULKAN_SDK}/lib)
	target_link_libraries(${app} shaderc_shared)
endif()

# copy models to executable

add_custom_command(TARGET ${app} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/resources/"
	COMMAND ${CMAKE_COMMAND} -E copy_directory 
		${MODELS_DIR}
		"$<TARGET_FILE_DIR:${app}>/resources"
)
