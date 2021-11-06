cmake_minimum_required(VERSION 3.18)

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
set(TINYOBJLOADER_DIR "${EXTERN_DIR}/tinyobjloader")
target_include_directories(${app} PRIVATE ${TINYOBJLOADER_DIR})

# volk
set(VOLK_DIR "${EXTERN_DIR}/volk")
add_subdirectory(${VOLK_DIR})
target_include_directories(${app} PRIVATE ${VOLK_DIR})
target_link_libraries(${app} volk)

# vulkan headers
set(VULKAN_HEADERS_DIR "${EXTERN_DIR}/Vulkan-Headers")
target_include_directories(${app} PRIVATE "${VULKAN_HEADERS_DIR}/include")

# vulkan sdk for shaderc
if (WIN32)
	target_link_libraries(${app} $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
else()
	target_link_directories(${app} PRIVATE $ENV{VULKAN_SDK}/lib)
	target_link_libraries(${app} shaderc_shared)
endif()

# download models
function(download_file url filepath hash_type hash)
if(NOT EXISTS ${filepath})
  get_filename_component(filename ${filepath} NAME_WE) 
  message("Downloading model: ${filename}")
 
  file(DOWNLOAD ${url} ${filepath}
	   SHOW_PROGRESS
       EXPECTED_HASH ${hash_type}=${hash})
endif()
endfunction(download_file)

download_file(
  https://casual-effects.com/g3d/data10/research/model/bmw/bmw.zip
  "${MODELS_DIR}/bmw.zip"
  SHA1 de6daa549019106df0512bc3f4dabec37d82e4b2)

file(ARCHIVE_EXTRACT 
	INPUT "${MODELS_DIR}/bmw.zip" 
	DESTINATION ${MODELS_DIR})

# copy models to executable resources
add_custom_command(TARGET ${app} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/resources/"
	COMMAND ${CMAKE_COMMAND} -E copy_directory 
		${MODELS_DIR}
		"$<TARGET_FILE_DIR:${app}>/resources"
)
