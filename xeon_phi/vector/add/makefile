# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -O2 -vec-report3 -masm=att -openmp add_fpd.c -o add_fpd
	icc -O2 -vec-report3 -masm=att -openmp add_int.c -o add_int
