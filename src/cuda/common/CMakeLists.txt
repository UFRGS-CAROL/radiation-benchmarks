cmake_minimum_required(VERSION 3.17)
project(common)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)
include_directories(src/jtx2/include)
include_directories(/usr/local/cuda/include)

add_executable(common
        include/cuda_utils.h
        include/device_vector.h
        include/generic_log.h
        include/JTX2Inst.h
        include/NVMLWrapper.h
        include/persistent_lib.h
        include/Profiler.h
        src/jtx2/example/sample.cpp
        src/jtx2/include/jtx1inst.h
        src/jtx2/src/jtx1pow.cpp
        src/jtx2/src/jtx1rate.cpp
        src/jtx2/src/jtx1temp.cpp
        src/JTX2Inst.cpp
        src/NVMLWrapper.cpp)
