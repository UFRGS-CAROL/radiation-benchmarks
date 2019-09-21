SOURCES_CHECK=hotspot_check.c ../../../../../include/log_helper.c
PRECISION = 32

CC = gcc
CC_FLAGS = -std=c99 -O0 -fopenmp

EXE_BASENAME=hotspot

NAME_SUFFIX=_not_hardened

all: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) hotspot.c -o $(EXE_BASENAME)_gen

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -o $(EXE_BASENAME)_check$(NAME_SUFFIX)

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o $(EXE_BASENAME)_err_inj$(NAME_SUFFIX)

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DTIMING -o $(EXE_BASENAME)_timing$(NAME_SUFFIX)

clean:
	rm -f hotspot_gen hotspot_check$(NAME_SUFFIX) hotspot_timing$(NAME_SUFFIX) hotspot_err_inj$(NAME_SUFFIX)
