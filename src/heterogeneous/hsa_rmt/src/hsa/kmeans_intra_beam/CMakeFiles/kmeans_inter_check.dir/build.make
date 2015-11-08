# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt

# Utility rule file for kmeans_inter_check.

# Include the progress variables for this target.
include src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/progress.make

src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check: src/hsa/kmeans_inter/main.cc
src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check: src/hsa/kmeans_inter/kmeans_benchmark.h
src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check: src/hsa/kmeans_inter/kmeans_benchmark.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Linting kmeans_inter_check"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter && /usr/bin/cmake -E chdir /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter cpplint.py main.cc kmeans_benchmark.h kmeans_benchmark.cc

kmeans_inter_check: src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check
kmeans_inter_check: src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/build.make
.PHONY : kmeans_inter_check

# Rule to build all files generated by this target.
src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/build: kmeans_inter_check
.PHONY : src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/build

src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/clean:
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter && $(CMAKE_COMMAND) -P CMakeFiles/kmeans_inter_check.dir/cmake_clean.cmake
.PHONY : src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/clean

src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/depend:
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/hsa/kmeans_inter/CMakeFiles/kmeans_inter_check.dir/depend

