# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-qopenmp

SOURCES_CHECK=hotspot_check.c ../../include/log_helper.c

general: hotspot hotspot_check hotspot_err_inj hotspot_timing

intel: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot 

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o hotspot_check

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o hotspot_err_inj

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o hotspot_timing

clean:
	rm -f hotspot hotspot_check hotspot_timing hotspot_err_inj
