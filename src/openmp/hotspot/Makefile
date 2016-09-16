# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -g -DMIC_NATIVE 
intel:OMP=-qopenmp

#all: hotspot hotspot_check
intel: hotspot hotspot_check

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot 

hotspot_check: hotspot_check.c
	$(CC) $(CC_FLAGS) hotspot_check.c  -DLOGS -o hotspot_check

clean:
	rm -f hotspot hotspot_check
