# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-qopenmp

SOURCES_CHECK=needle_check.cpp ../../include/log_helper.c
general:SOURCES_CHECK=needle_check.c ../../include/log_helper.c

general: needle_gen needle_check needle_err_inj needle_timing

intel: needle_gen needle_check needle_err_inj needle_timing

needle_gen: needle_gen.cpp
	$(CC) $(CC_FLAGS) needle_gen.cpp -o needle_gen 

needle_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o needle_check

needle_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o needle_err_inj

needle_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o needle_timing

clean:
	rm -f needle_gen needle_check needle_err_inj needle_timing

