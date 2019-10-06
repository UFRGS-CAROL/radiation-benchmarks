# C compiler
CC = gcc
CC_FLAGS = -std=c99 -O0 -fopenmp

SOURCES_CHECK=mergesort_check.c /home/carol/radiation-benchmarks/src/include/log_helper.c /home/carol/radiation-benchmarks/src/openmp/selective_hardening/src/selective_hardening.c

NAME_SUFFIX=_hardened_2

general: merge_gen merge_check merge_err_inj merge_timing

merge_gen: mergesort_gen.c generateInput.c
	$(CC) $(CC_FLAGS) mergesort_gen.c -o merge_gen
	$(CC) $(CC_FLAGS) generateInput.c -o generateInput

merge_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o merge_check$(NAME_SUFFIX)

merge_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o merge_err_inj$(NAME_SUFFIX)

merge_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o merge_timing$(NAME_SUFFIX)

clean:
	rm -f generateInput merge_gen merge_check$(NAME_SUFFIX) merge_err_inj$(NAME_SUFFIX) merge_timing$(NAME_SUFFIX)
