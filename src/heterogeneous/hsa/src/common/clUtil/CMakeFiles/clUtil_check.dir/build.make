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
CMAKE_SOURCE_DIR = /home/carol/HSA_BENCH/Hetero-Mark

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/carol/HSA_BENCH/Hetero-Mark

# Utility rule file for clUtil_check.

# Include the progress variables for this target.
include src/common/clUtil/CMakeFiles/clUtil_check.dir/progress.make

src/common/clUtil/CMakeFiles/clUtil_check: src/common/clUtil/clError.h
src/common/clUtil/CMakeFiles/clUtil_check: src/common/clUtil/clFile.h
src/common/clUtil/CMakeFiles/clUtil_check: src/common/clUtil/clProfiler.h
src/common/clUtil/CMakeFiles/clUtil_check: src/common/clUtil/clRuntime.h
src/common/clUtil/CMakeFiles/clUtil_check: src/common/clUtil/clUtil.h
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/HSA_BENCH/Hetero-Mark/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Linting clUtil_check"
	cd /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil && /usr/bin/cmake -E chdir /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil cpplint.py clError.h clFile.h clProfiler.h clRuntime.h clUtil.h

clUtil_check: src/common/clUtil/CMakeFiles/clUtil_check
clUtil_check: src/common/clUtil/CMakeFiles/clUtil_check.dir/build.make
.PHONY : clUtil_check

# Rule to build all files generated by this target.
src/common/clUtil/CMakeFiles/clUtil_check.dir/build: clUtil_check
.PHONY : src/common/clUtil/CMakeFiles/clUtil_check.dir/build

src/common/clUtil/CMakeFiles/clUtil_check.dir/clean:
	cd /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil && $(CMAKE_COMMAND) -P CMakeFiles/clUtil_check.dir/cmake_clean.cmake
.PHONY : src/common/clUtil/CMakeFiles/clUtil_check.dir/clean

src/common/clUtil/CMakeFiles/clUtil_check.dir/depend:
	cd /home/carol/HSA_BENCH/Hetero-Mark && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/carol/HSA_BENCH/Hetero-Mark /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil /home/carol/HSA_BENCH/Hetero-Mark /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil /home/carol/HSA_BENCH/Hetero-Mark/src/common/clUtil/CMakeFiles/clUtil_check.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/common/clUtil/CMakeFiles/clUtil_check.dir/depend

