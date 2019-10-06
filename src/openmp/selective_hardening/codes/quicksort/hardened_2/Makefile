# C compiler
CC = gcc
CC_FLAGS = -std=c99 -O0 -fopenmp

SOURCES_CHECK=quicksort_check.c /home/carol/radiation-benchmarks/src/include/log_helper.c /home/carol/radiation-benchmarks/src/openmp/selective_hardening/src/selective_hardening.c

NAME_SUFFIX=_hardened_2

general: quick_gen quick_check quick_err_inj quick_timing

quick_gen: quicksort_gen.c generateInput.c
	$(CC) $(CC_FLAGS) quicksort_gen.c -o quick_gen
	$(CC) $(CC_FLAGS) generateInput.c -o genInput

quick_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o quick_check$(NAME_SUFFIX)

quick_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o quick_err_inj$(NAME_SUFFIX)

quick_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o quick_timing$(NAME_SUFFIX)

clean:
	rm -f genInput quick_gen quick_check* quick_err_inj* quick_timing*
