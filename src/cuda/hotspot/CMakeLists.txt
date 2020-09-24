cmake_minimum_required(VERSION 3.17)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(hotspot CUDA)

set(CMAKE_CXX_STANDARD 14)

include_directories(.
        /usr/local/cuda/include
        /usr/local/cuda/samples/common/inc/
        ../common/include
        )


add_executable(hotspot
        cuda_hotspot.cu
        half.hpp)
