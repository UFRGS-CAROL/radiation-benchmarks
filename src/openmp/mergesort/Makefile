# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

SOURCES_CHECK=mergesort_check.c ../../include/log_helper.c
general:SOURCES_CHECK=mergesort_check.c ../../include/log_helper.c

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

armv8: merge_gen merge_check merge_err_inj merge_timing

general: merge_gen merge_check merge_err_inj merge_timing

intel: merge_gen merge_check merge_err_inj merge_timing

merge_gen: mergesort_gen.c generateInput.c
	$(CC) $(CC_FLAGS) mergesort_gen.c -o merge_gen
	$(CC) $(CC_FLAGS) generateInput.c -o merge_genInput

merge_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o merge_check

merge_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o merge_err_inj

merge_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o merge_timing

clean:
	rm -f genInput merge_gen merge_check merge_err_inj merge_timing
