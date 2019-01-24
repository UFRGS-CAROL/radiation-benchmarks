# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -DBOFFSET=12

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp -DBOFFSET=12

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp

armv8:dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices

general: dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices

intel: dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices

dgemm_gen: dgemm_gen.c ../../include/log_helper.c
	$(CC) ../../include/log_helper.c dgemm_gen.c -o dgemm_gen $(CC_FLAGS)

dgemm_check: dgemm.c ../../include/log_helper.c
	$(CC) ../../include/log_helper.c dgemm.c -DLOGS -o dgemm_check $(CC_FLAGS)

dgemm_err_inj: dgemm.c  ../../include/log_helper.c
	$(CC) ../../include/log_helper.c dgemm.c -DLOGS -DERR_INJ -o dgemm_err_inj $(CC_FLAGS)

dgemm_timing: dgemm.c ../../include/log_helper.c
	$(CC) ../../include/log_helper.c dgemm.c -DTIMING -o dgemm_timing $(CC_FLAGS)

genMatrices: generateMatrices.cpp
	g++ generateMatrices.cpp -o genMatrices

clean:
	rm -f dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices
