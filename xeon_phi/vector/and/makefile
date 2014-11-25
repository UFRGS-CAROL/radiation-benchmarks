# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp -no-vec ../../../include/log_helper.c and_int.c -o and_int
	icc -O2 -vec-report3 -masm=att -openmp -no-vec ../../../include/log_helper.c and_fpd.c -o and_fpd
