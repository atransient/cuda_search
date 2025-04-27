cmake_minimum_required(VERSION 3.17)

project(cuda_learn LANGUAGES CXX CUDA)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# set(CMAKE_CUDA_ARCHITECTURES "60;75")
# set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CUDA_ARCHITECTURES "90")

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/utils")

message(STATUS "include_path : ${include_path}")

include_directories(${include_path})

add_subdirectory(cuda_basic_usage)
add_subdirectory(smem_related)