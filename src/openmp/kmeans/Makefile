# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

SOURCES_AUX=cluster.c getopt.c kmeans_clustering.c ../../include/log_helper.c
general:SOURCES_CHECK=quicksort_check.c ../../include/log_helper.c

armv8: kmeans_gen kmeans_check

general: kmeans_gen kmeans_check

intel: kmeans_gen kmeans_check

kmeans_gen: $(SOURCES_AUX) kmeans_gen.c
	$(CC)  $(CC_FLAGS) $(SOURCES_AUX) kmeans_gen.c -o kmeans_gen

kmeans_check: $(SOURCES_AUX) kmeans_check.c
	$(CC)  $(CC_FLAGS) $(SOURCES_AUX)  kmeans_check.c -DLOGS -o kmeans_check

clean:
	rm -f *.o *~ kmeans_gen kmeans_check
