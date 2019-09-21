# C compiler
CC = gcc
CC_FLAGS = -std=c99 -O0 -fopenmp

SOURCES_CHECK=mergesort_check.c ../../../../../include/log_helper.c

NAME_SUFFIX=_not_hardened

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
