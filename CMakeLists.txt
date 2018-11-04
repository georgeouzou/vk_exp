cmake_minimum_required(VERSION 3.7)

set(app VulkanExperiments)

project(${app})

set(src src/main.cpp)

add_executable(${app} ${src})
