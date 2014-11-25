# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp -no-vec ../../../include/log_helper.c slr_int.c -o slr_int
	icc -O2 -vec-report3 -masm=att -openmp -no-vec ../../../include/log_helper.c slr_fpd.c -o slr_fpd
