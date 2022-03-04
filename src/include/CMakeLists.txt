cmake_minimum_required(VERSION 3.21)
project(LogHelper LANGUAGES C VERSION 1.0.0 DESCRIPTION "Log Helper for radiation experiments")

set(CMAKE_C_STANDARD 11)

add_library(LogHelper SHARED log_helper.c)

set_target_properties(LogHelper PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(LogHelper PROPERTIES PUBLIC_HEADER log_helper.h)

target_include_directories(LogHelper PRIVATE .)

include(GNUInstallDirs)

install(TARGETS LogHelper
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


add_subdirectory(example_log)