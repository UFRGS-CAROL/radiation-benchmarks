SOURCES_CHECK=hotspot_check.c ../../include/log_helper.c
PRECISION ?= 64
ARCH?=generic


# General configuration
ifeq ($(ARCH), generic)
CC = gcc
CC_FLAGS = -fopenmp -O3

# Intel Xeon Phi KNC configuration
else ifeq ($(ARCH), knc)
CC=icc -mmic -DMIC_NATIVE 
CC_FLAGS=-qopenmp

# ThunderX2 Configuration
else ifeq ($(ARCH), thunderx2)
CC = gcc
# Suggested flags for thunderxs taken from prace 'Best Practice Guide - ARM64'
CC_FLAGS=-Ofast -march=armv8.1-a -mcpu=thunderx2t99 -mtune=thunderx2t99 -fopenmp -funroll-loops

endif

EXE_BASENAME=hotspot_$(ARCH)_$(PRECISION)

all: hotspot hotspot_check hotspot_err_inj hotspot_timing

hotspot: hotspot.c
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) hotspot.c -o $(EXE_BASENAME)_gen 

hotspot_check: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -o $(EXE_BASENAME)_check

hotspot_err_inj: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DLOGS -DERR_INJ -o $(EXE_BASENAME)_err_inj

hotspot_timing: $(SOURCES_CHECK)
	$(CC) $(CC_FLAGS) -DFP=$(PRECISION) $(SOURCES_CHECK) -DTIMING -o $(EXE_BASENAME)_timing

clean:
	rm -f hotspot*gen hotspot*check hotspot*timing hotspot*err_inj

# Display help message
help:
	@echo "make options: 'ARCH' and 'PRECISION'"
	@echo "ARCH: 'generic', 'knc', 'thunderx2'"
	@echo "PRECISION: '32', '64'"
	@echo "Default: ARCH=generic PRECISION=64"
	@echo ""
	@echo "Compiling for thunderx2 using single precision: make ARCH=thunderx2 PRECISION=32"
	@echo "Compiling for knc using double precision: make ARCH=knc PRECISION=64"
