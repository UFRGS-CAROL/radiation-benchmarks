# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -lm

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

SOURCES_CHECK=lud_check.c lud_omp.c ../../include/log_helper.c

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

armv8: lud_gen lud_check lud_err_inj lud_timing

general: lud_gen lud_check lud_err_inj lud_timing

intel: lud_gen lud_check lud_err_inj lud_timing

lud_gen: lud_gen.c common.c
	$(CC) common.c lud_omp.c lud_gen.c -o lud_gen  $(CC_FLAGS)

lud_check: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DLOGS -o lud_check $(CC_FLAGS)

lud_err_inj: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DLOGS -DERR_INJ -o lud_err_inj $(CC_FLAGS)

lud_timing: $(SOURCES_CHECK)
	$(CC) common.c $(SOURCES_CHECK) -DTIMING -o lud_timing $(CC_FLAGS)

clean:
	rm -f lud_gen lud_check lud_err_inj lud_timing
