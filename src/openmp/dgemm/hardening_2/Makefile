# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -DBOFFSET=12

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp -DBOFFSET=12


general: dgemm_gen dgemm_check dgemm_check_hardened dgemm_err_inj dgemm_timing dgemm_timing_hardened genMatrices

intel: dgemm_gen dgemm_check dgemm_check_hardened dgemm_err_inj dgemm_timing dgemm_timing_hardened genMatrices

dgemm_gen: dgemm_gen.c ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm_gen.c -o dgemm_gen $(CC_FLAGS)

dgemm_check: dgemm.c ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm.c -DLOGS -o dgemm_check $(CC_FLAGS)

dgemm_check_hardened: dgemm_hardened.c ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm_hardened.c -DLOGS -o dgemm_check_hardened_2 $(CC_FLAGS)

dgemm_err_inj: dgemm.c  ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm.c -DLOGS -DERR_INJ -o dgemm_err_inj $(CC_FLAGS)

dgemm_timing: dgemm.c ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm.c -DTIMING -o dgemm_timing $(CC_FLAGS)

dgemm_timing_hardened: dgemm_hardened.c ../../../include/log_helper.c
	$(CC) ../../../include/log_helper.c dgemm_hardened.c -DTIMING -o dgemm_timing_hardened_2 $(CC_FLAGS)

genMatrices: generateMatrices.cpp
	g++ generateMatrices.cpp -o genMatrices

clean:
	rm -f dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices dgemm_check_hardened_2 dgemm_timing_hardened_2
