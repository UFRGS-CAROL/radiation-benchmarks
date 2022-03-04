cmake_minimum_required(VERSION 3.21)
project(example_log DESCRIPTION "examples how to use LogHelper lib")

set(CMAKE_BUILD_TYPE Release)


add_executable(example_log example_log.c)

include_directories(example_log ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(example_log ${CMAKE_SOURCE_DIR}/build/libLogHelper.so)
link_directories(${CMAKE_SOURCE_DIR}/build)