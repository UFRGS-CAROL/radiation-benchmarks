# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp fma_fpd.c -o fma_fpd
	icc -O2 -vec-report3 -masm=att -openmp fma_int.c -o fma_int
