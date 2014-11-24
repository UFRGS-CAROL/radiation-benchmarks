# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp mul_fpd.c -o mul_fpd
	icc -O2 -vec-report3 -masm=att -openmp mul_int.c -o mul_int
