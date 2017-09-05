# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3 -lm

intel:CC=icc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-qopenmp

SOURCES_AUX=../../include/log_helper.c

general: nn_gen nn_check

intel: nn_gen nn_check

nn_gen: $(SOURCES_AUX) nn_gen.c
	$(CC)  $(CC_FLAGS) $(SOURCES_AUX) nn_gen.c -o nn_gen
 
nn_check: $(SOURCES_AUX) nn_check.c 
	$(CC)  $(CC_FLAGS) $(SOURCES_AUX)  nn_check.c -DLOGS -o nn_check
 
clean:
	rm -f *.o *~ nn_gen nn_check
