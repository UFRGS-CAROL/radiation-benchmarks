cmake_minimum_required(VERSION 3.17)
project(cache)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

include_directories(../common/include)
include_directories(/usr/local/cuda/include)


add_executable(cache
        src/CacheLine.h
        src/cupti_profiler.h
        src/l1_cache_kernels.cu
        src/L1Cache.h
        src/l2_cache_kernels.cu
        src/L2Cache.h
        src/Log.h
        src/main.cpp
        src/Memory.h
        src/Parameters.cpp
        src/Parameters.h
        src/read_only_cache_kernels.cu
        src/ReadOnly.h
        src/register_file_kernels.cu
        src/register_kernel.h
        src/register_kernel_volta.h
        src/RegisterFile.h
        src/shared_memory_kernels.cu
        src/SharedMemory.h
        src/utils.h
        src/device_functions.h
        )
