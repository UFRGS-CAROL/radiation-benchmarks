# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3
LOGS=1

intel:CC=icc -mmic -DMIC_NATIVE
intel:CC_FLAGS=-qopenmp

armv8:CC=aarch64-linux-gnu-gcc
armv8:CC_FLAGS=-O3 -static -fopenmp -lm

ifeq ($(LOGS), 1)
LIBRARY = -L../../include -I../../include -lLogHelper -DLOGS
endif



SOURCES_CHECK=hotspot_check.c

armv8:hotspot hotspot_check hotspot_err_inj hotspot_timing

general: hotspot hotspot_check hotspot_err_inj hotspot_timing

intel: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot_gen

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -o hotspot_check -lm  $(LIBRARY)

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DERR_INJ -o hotspot_err_inj  $(LIBRARY)

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o hotspot_timing

clean:
	rm -f hotspot_gen hotspot_check hotspot_timing hotspot_err_inj
