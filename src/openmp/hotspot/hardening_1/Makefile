# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O3

intel:CC=icc -mmic -DMIC_NATIVE 
intel:CC_FLAGS=-qopenmp

SOURCES_CHECK=hotspot_check.c ../../../include/log_helper.c
SOURCES_CHECK_HARDENED=hotspot_check_hardened.c ../../../include/log_helper.c

general: hotspot hotspot_check hotspot_check_hardened hotspot_err_inj hotspot_timing hotspot_timing_hardened

intel: hotspot hotspot_check hotspot_check_hardened hotspot_err_inj hotspot_timing hotspot_timing_hardened

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) hotspot.c -o hotspot_gen 

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -o hotspot_check

hotspot_check_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK_HARDENED) -DLOGS -o hotspot_check_hardened

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o hotspot_err_inj

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK) -DTIMING -o hotspot_timing

hotspot_timing_hardened: $(SOURCES_CHECK_HARDENED)
	$(CC) $(CC_FLAGS) $(SOURCES_CHECK_HARDENED) -DTIMING -o hotspot_timing_hardened

clean:
	rm -f hotspot_gen hotspot_check hotspot_check_hardened hotspot_timing hotspot_timing_hardened hotspot_err_inj
