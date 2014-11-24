# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp div_fpd.c -o div_fpd
	icc -O2 -vec-report3 -masm=att -openmp div_int.c -o div_int
