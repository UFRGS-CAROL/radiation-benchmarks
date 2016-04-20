# make sure you source the compilervars of intel
#   source /opt/intel/composerxe/bin/compilervars.sh intel64;
all:
	icc -offload-attribute-target=mic -O2 -vec-report3 -masm=att -openmp -no-vec ../../../../include/log_helper.c or_int.c -o or_int
	icc -offload-attribute-target=mic -O2 -vec-report3 -masm=att -openmp -no-vec ../../../../include/log_helper.c or_fpd.c -o or_fpd
	icc -offload-attribute-target=mic -DALL_DEBUG=1 -O2 -vec-report3 -masm=att -openmp -no-vec ../../../../include/log_helper.c or_int.c -o or_int_debug
	icc -offload-attribute-target=mic -DALL_DEBUG=1 -O2 -vec-report3 -masm=att -openmp -no-vec ../../../../include/log_helper.c or_fpd.c -o or_fpd_debug
