# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

SOURCES_CHECK=hotspot_check.c ../../include/log_helper.c

armv8:hotspot hotspot_check hotspot_err_inj hotspot_timing

general: hotspot hotspot_check hotspot_err_inj hotspot_timing

intel: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot_gen

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o hotspot_check -lm

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o hotspot_err_inj

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o hotspot_timing

clean:
	rm -f hotspot_gen hotspot_check hotspot_timing hotspot_err_inj
