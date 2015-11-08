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

# Include any dependencies generated for this target.
include src/common/command_line_option/CMakeFiles/command_line_option.dir/depend.make

# Include the progress variables for this target.
include src/common/command_line_option/CMakeFiles/command_line_option.dir/progress.make

# Include the compile flags for this target's objects.
include src/common/command_line_option/CMakeFiles/command_line_option.dir/flags.make

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o: src/common/command_line_option/CMakeFiles/command_line_option.dir/flags.make
src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o: src/common/command_line_option/command_line_option.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/command_line_option.dir/command_line_option.cc.o -c /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/command_line_option.cc

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/command_line_option.dir/command_line_option.cc.i"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/command_line_option.cc > CMakeFiles/command_line_option.dir/command_line_option.cc.i

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/command_line_option.dir/command_line_option.cc.s"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/command_line_option.cc -o CMakeFiles/command_line_option.dir/command_line_option.cc.s

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.requires:
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.requires

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.provides: src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.requires
	$(MAKE) -f src/common/command_line_option/CMakeFiles/command_line_option.dir/build.make src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.provides.build
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.provides

src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.provides.build: src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o: src/common/command_line_option/CMakeFiles/command_line_option.dir/flags.make
src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o: src/common/command_line_option/option_parser_impl.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/command_line_option.dir/option_parser_impl.cc.o -c /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_parser_impl.cc

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/command_line_option.dir/option_parser_impl.cc.i"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_parser_impl.cc > CMakeFiles/command_line_option.dir/option_parser_impl.cc.i

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/command_line_option.dir/option_parser_impl.cc.s"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_parser_impl.cc -o CMakeFiles/command_line_option.dir/option_parser_impl.cc.s

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.requires:
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.requires

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.provides: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.requires
	$(MAKE) -f src/common/command_line_option/CMakeFiles/command_line_option.dir/build.make src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.provides.build
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.provides

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.provides.build: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o: src/common/command_line_option/CMakeFiles/command_line_option.dir/flags.make
src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o: src/common/command_line_option/option_setting_help_printer.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o -c /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_help_printer.cc

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.i"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_help_printer.cc > CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.i

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.s"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_help_printer.cc -o CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.s

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.requires:
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.requires

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.provides: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.requires
	$(MAKE) -f src/common/command_line_option/CMakeFiles/command_line_option.dir/build.make src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.provides.build
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.provides

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.provides.build: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o: src/common/command_line_option/CMakeFiles/command_line_option.dir/flags.make
src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o: src/common/command_line_option/option_setting_impl.cc
	$(CMAKE_COMMAND) -E cmake_progress_report /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/command_line_option.dir/option_setting_impl.cc.o -c /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_impl.cc

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/command_line_option.dir/option_setting_impl.cc.i"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_impl.cc > CMakeFiles/command_line_option.dir/option_setting_impl.cc.i

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/command_line_option.dir/option_setting_impl.cc.s"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/option_setting_impl.cc -o CMakeFiles/command_line_option.dir/option_setting_impl.cc.s

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.requires:
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.requires

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.provides: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.requires
	$(MAKE) -f src/common/command_line_option/CMakeFiles/command_line_option.dir/build.make src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.provides.build
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.provides

src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.provides.build: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o

# Object files for target command_line_option
command_line_option_OBJECTS = \
"CMakeFiles/command_line_option.dir/command_line_option.cc.o" \
"CMakeFiles/command_line_option.dir/option_parser_impl.cc.o" \
"CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o" \
"CMakeFiles/command_line_option.dir/option_setting_impl.cc.o"

# External object files for target command_line_option
command_line_option_EXTERNAL_OBJECTS =

src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o
src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o
src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o
src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o
src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/build.make
src/common/command_line_option/libcommand_line_option.a: src/common/command_line_option/CMakeFiles/command_line_option.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libcommand_line_option.a"
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && $(CMAKE_COMMAND) -P CMakeFiles/command_line_option.dir/cmake_clean_target.cmake
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/command_line_option.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/common/command_line_option/CMakeFiles/command_line_option.dir/build: src/common/command_line_option/libcommand_line_option.a
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/build

src/common/command_line_option/CMakeFiles/command_line_option.dir/requires: src/common/command_line_option/CMakeFiles/command_line_option.dir/command_line_option.cc.o.requires
src/common/command_line_option/CMakeFiles/command_line_option.dir/requires: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_parser_impl.cc.o.requires
src/common/command_line_option/CMakeFiles/command_line_option.dir/requires: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_help_printer.cc.o.requires
src/common/command_line_option/CMakeFiles/command_line_option.dir/requires: src/common/command_line_option/CMakeFiles/command_line_option.dir/option_setting_impl.cc.o.requires
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/requires

src/common/command_line_option/CMakeFiles/command_line_option.dir/clean:
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option && $(CMAKE_COMMAND) -P CMakeFiles/command_line_option.dir/cmake_clean.cmake
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/clean

src/common/command_line_option/CMakeFiles/command_line_option.dir/depend:
	cd /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option /home/carol/vinicius/radiation-benchmarks/src/heterogeneous/hsa_rmt/src/common/command_line_option/CMakeFiles/command_line_option.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/common/command_line_option/CMakeFiles/command_line_option.dir/depend

