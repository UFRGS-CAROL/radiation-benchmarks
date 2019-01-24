# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp

arm:CC=arm-linux-gnueabihf-gcc
arm:CC_FLAGS=-mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard -fopenmp -static

SOURCES_CHECK=quicksort_check.c ../../include/log_helper.c
general:SOURCES_CHECK=quicksort_check.c ../../include/log_helper.c

arm: quick_gen quick_check quick_err_inj quick_timing

armv8: quick_gen quick_check quick_err_inj quick_timing

general: quick_gen quick_check quick_err_inj quick_timing

intel: quick_gen quick_check quick_err_inj quick_timing

quick_gen: quicksort_gen.c generateInput.c
	$(CC) $(CC_FLAGS) quicksort_gen.c -o quick_gen
	$(CC) $(CC_FLAGS) generateInput.c -o genInput

quick_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o quick_check

quick_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o quick_err_inj

quick_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o quick_timing

clean:
	rm -f genInput quick_gen quick_check quick_err_inj quick_timing
