cmake_minimum_required(VERSION 3.7)

set(app VulkanExperiments)

project(${app})

set(EXTERN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern")
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

set(src 
	src/main.cpp 
	src/stb_image.c
	src/tiny_obj_loader.cpp
)

add_executable(${app} ${src})
set_property(TARGET ${app} PROPERTY CXX_STANDARD 17)

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
#target_link_libraries(${app} ${Vulkan_LIBRARIES})
target_include_directories(${app} PRIVATE ${Vulkan_INCLUDE_DIRS})

if (WIN32)
	target_compile_definitions(${app} PRIVATE "VK_USE_PLATFORM_WIN32_KHR")
	target_compile_definitions(${app} PRIVATE "GLFW_EXPOSE_NATIVE_WIN32")
endif()

# compile shaders

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
	set(GLSL_VALIDATOR "$ENV{VULKAN_SDK}/Bin/glslangValidator.exe")
else()
	message(FATAL_ERROR "32-bit systems are not supported!")
endif()

file(GLOB_RECURSE GLSL_SRC_FILES
	"${SHADER_DIR}/*.frag"
	"${SHADER_DIR}/*.vert"
	"${SHADER_DIR}/*.rgen"
	"${SHADER_DIR}/*.rchit"
	"${SHADER_DIR}/*.rmiss"
)

foreach(GLSL ${GLSL_SRC_FILES})
	get_filename_component(FILE_NAME ${GLSL} NAME)
	set(SPIRV "${PROJECT_BINARY_DIR}/spirv_files/${FILE_NAME}.spv")
	add_custom_command(
		OUTPUT ${SPIRV}
		COMMAND ${CMAKE_COMMAND} -E make_directory "${PROJECT_BINARY_DIR}/spirv_files/"
		COMMAND ${GLSL_VALIDATOR} -V  ${GLSL} -o ${SPIRV}
		DEPENDS ${GLSL}
	)
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
	Shaders
	DEPENDS ${SPIRV_BINARY_FILES}
)

add_dependencies(${app} Shaders)

# download resources

set(DOWNLOADABLES "${PROJECT_BINARY_DIR}/downloadables/")
file(MAKE_DIRECTORY ${DOWNLOADABLES})

if (NOT EXISTS "${DOWNLOADABLES}/chalet.obj")
	MESSAGE(STATUS "Downloading model resources")
	file(DOWNLOAD https://vulkan-tutorial.com/resources/chalet.obj.zip "${DOWNLOADABLES}/chalet.obj.zip"  SHOW_PROGRESS)
	execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${DOWNLOADABLES}/chalet.obj.zip WORKING_DIRECTORY ${DOWNLOADABLES})
	file(REMOVE ${DOWNLOADABLES}/chalet.obj.zip)
endif()
if (NOT EXISTS "${DOWNLOADABLES}/chalet.jpg")
	file(DOWNLOAD https://vulkan-tutorial.com/resources/chalet.jpg "${DOWNLOADABLES}/chalet.jpg"  SHOW_PROGRESS)
endif()

# copy shaders to executable

add_custom_command(TARGET ${app} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/resources/"
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${PROJECT_BINARY_DIR}/spirv_files"
		"$<TARGET_FILE_DIR:${app}>/resources"
)

# copy images to executable

add_custom_command(TARGET ${app} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/resources/"
	COMMAND ${CMAKE_COMMAND} -E copy_directory 
		${DOWNLOADABLES}
		"$<TARGET_FILE_DIR:${app}>/resources"
)
