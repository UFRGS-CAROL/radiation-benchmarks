cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(accl CUDA)

#set(CMAKE_CUDA_STANDARD /usr/local/cuda)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++")

include_directories(src)
include_directories(/usr/local/cuda/include)
include_directories(../common/include)

set(CMAKE_CUDA_ARCHITECTURES 35 70)

add_executable(accl
        src/accl.cu
        src/accl.h
        src/image.h
        src/main.cpp
        src/misc.h
        src/Parameters.cpp
        src/Parameters.h
        src/utils.h)

set_target_properties(
        accl
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)