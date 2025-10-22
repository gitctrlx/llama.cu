cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_BUILD_TYPE Debug)

project(llama.cu LANGUAGES C CUDA)

add_definitions(-DUSE_CUBLAS=1)

add_executable(llama llama.cu)
target_link_libraries(llama -lm -lcublas)
