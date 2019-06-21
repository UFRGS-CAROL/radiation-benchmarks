SOURCES_CHECK=lud_check.c lud_omp.c ../../include/log_helper.c
PRECISION ?= 64                
ARCH?=generic

# General configuration
ifeq ($(ARCH), generic)
CC = gcc
CC_FLAGS = -fopenmp -O3 -lm

# Intel Xeon Phi KNC configuration
else ifeq ($(ARCH), knc)
CC=icc -mmic -DMIC_NATIVE
CC_FLAGS=-qopenmp

# ThunderX2 Configuration
else ifeq ($(ARCH), thunderx2)
CC = gcc
# Suggested flags for thunderxs taken from prace 'Best Practice Guide - ARM64'
CC_FLAGS=-Ofast -march=armv8.1-a -mcpu=thunderx2t99 -mtune=thunderx2t99 -fopenmp -funroll-loops -lm

endif


EXE_BASENAME=lud_$(ARCH)_$(PRECISION)

all: lud_gen lud_check lud_err_inj lud_timing

lud_gen: lud_gen.c common.c
	$(CC) -DPRECISION=$(PRECISION) common.c lud_omp.c lud_gen.c -o $(EXE_BASENAME)_gen  $(CC_FLAGS)

lud_check: $(SOURCES_CHECK)
	$(CC) -DPRECISION=$(PRECISION) common.c $(SOURCES_CHECK) -DLOGS -o $(EXE_BASENAME)_check $(CC_FLAGS)

lud_err_inj: $(SOURCES_CHECK)
	$(CC) -DPRECISION=$(PRECISION) common.c $(SOURCES_CHECK) -DLOGS -DERR_INJ -o $(EXE_BASENAME)_err_inj $(CC_FLAGS)

lud_timing: $(SOURCES_CHECK)
	$(CC) -DPRECISION=$(PRECISION) common.c $(SOURCES_CHECK) -DTIMING -o $(EXE_BASENAME)_timing $(CC_FLAGS)

clean:
	rm -f lud*gen lud*check lud*err_inj lud*timing

# Display help message
help:
	@echo "make options: 'ARCH' and 'PRECISION'"
	@echo "ARCH: 'generic', 'knc', 'thunderx2'"
	@echo "PRECISION: '32', '64'"
	@echo "Default: ARCH=generic PRECISION=64"
	@echo ""
	@echo "Compiling for thunderx2 using single precision: make ARCH=thunderx2 PRECISION=32"
	@echo "Compiling for knc using double precision: make ARCH=knc PRECISION=64"
