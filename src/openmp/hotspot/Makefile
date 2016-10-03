# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -g -DMIC_NATIVE 
intel:OMP=-qopenmp

SOURCES_CHECK=hotspot_check.c ../../include/log_helper.c

#all: hotspot hotspot_check
intel: hotspot hotspot_check

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot 

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o hotspot_check

clean:
	rm -f hotspot hotspot_check
