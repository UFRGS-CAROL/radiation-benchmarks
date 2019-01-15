# C compiler
CC = gcc
CXX = g++
DEFAULT_INPUT?=8192
LOGS?=1
PRECISION?=float

#Three archs
# x86 arm xeon
ARCH?=x86
EXEC=gemm_$(PRECISION)

#default is float
CC_FLAGS = -O3 -DDEFAULT_INPUT_SIZE=$(DEFAULT_INPUT)

ifeq ($(PRECISION), float)
CC_FLAGS+= -DPRECISION=32
endif

ifeq ($(PRECISION), double)
CC_FLAGS+= -DPRECISION=64
endif

ifeq ($(ARCH), xeon) 
CC_FLAGS+= -qopenmp -DBOFFSET=12 icc -mmic -DMIC_NATIVE

else ifeq ($(ARCH), arm)
CC_FLAGS+= -fopenmp -DBOFFSET=12 -static -mfloat-abi=hard -mfpu=auto

else ifeq ($(ARCH), x86)
CC_FLAGS+= -fopenmp -DBOFFSET=12

endif

ifeq ($(LOGS), 1)
INCLUDE+= -I../../include
LIBRARY+= -L../../include -lLogHelper -DLOGS
endif


all: $(EXEC)_gen $(EXEC)_check $(EXEC)_err_inj $(EXEC)_timing 

$(EXEC)_gen: gemm_gen.c 
	$(CC) $(CC_FLAGS)  gemm_gen.c -o $(EXEC)_gen $(INCLUDE) $(LIBRARY) 

$(EXEC)_check: gemm.c 
	$(CC) $(CC_FLAGS) gemm.c -o $(EXEC)_check $(INCLUDE) $(LIBRARY)

$(EXEC)_err_inj: gemm.c  
	$(CC) $(CC_FLAGS) gemm.c -DERR_INJ -o $(EXEC)_err_inj $(INCLUDE) $(LIBRARY)

$(EXEC)_timing: gemm.c
	$(CC) $(CC_FLAGS) gemm.c -DTIMING -o $(EXEC)_timing $(INCLUDE) $(LIBRARY)

generate:
	./$(EXEC)_gen 4 1024 8 /tmp/testa /tmp/testb /tmp/gold

test:
	./$(EXEC)_check 4 1024 8 /tmp/testa /tmp/testb /tmp/gold 10

clean:
	rm -f gemm_float_gen gemm_float_check gemm_float_err_inj gemm_float_timing gemm_double_err_inj  gemm_double_gen  gemm_double_timing
