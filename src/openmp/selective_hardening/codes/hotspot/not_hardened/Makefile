SOURCES_CHECK=hotspot_check.c ../../../../../include/log_helper.c
PRECISION = 32

CC = gcc
CC_FLAGS = -std=c99 -g -O0 -fopenmp

EXE_BASENAME=hotspot

NAME_SUFFIX=_not_hardened

all: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) hotspot.c -o $(EXE_BASENAME)_gen$(NAME_SUFFIX)

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -o $(EXE_BASENAME)_check$(NAME_SUFFIX)

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o $(EXE_BASENAME)_err_inj$(NAME_SUFFIX)

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DTIMING -o $(EXE_BASENAME)_timing$(NAME_SUFFIX)

clean:
	rm -f hotspot_gen_not_hardened hotspot_check_not_hardened hotspot_timing_not_hardened hotspot_err_inj_not_hardened
