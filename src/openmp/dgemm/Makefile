# C compiler
CC = gcc
CXX = g++
DEFAULT_INPUT?=8192
LOGS?=1

#Three archs
# x86 arm xeon
ARCH?=x86

#Default flags
CC_FLAGS = -O3 


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

all: dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices

dgemm_gen: dgemm_gen.c ../../include/log_helper.c
	$(CC)  dgemm_gen.c -o dgemm_gen $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

dgemm_check: dgemm.c ../../include/log_helper.c
	$(CC)  dgemm.c -o dgemm_check $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

dgemm_err_inj: dgemm.c  ../../include/log_helper.c
	$(CC) dgemm.c -DERR_INJ -o dgemm_err_inj $(CC_FLAGS)  $(INCLUDE) $(LIBRARY)

dgemm_timing: dgemm.c ../../include/log_helper.c
	$(CC)  dgemm.c -DTIMING -o dgemm_timing $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

genMatrices: generateMatrices.cpp
	$(CXX) generateMatrices.cpp -o genMatrices -DDEFAULT_INPUT_SIZE=$(DEFAULT_INPUT)


generate:
	./genMatrices /tmp/testa /tmp/testb
	./dgemm_gen 4 1024 8 /tmp/testa /tmp/testb /tmp/gold

test:
	./dgemm_check 4 1024 8 /tmp/testa /tmp/testb /tmp/gold 10

clean:
	rm -f dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices
