# C compiler
CC = gcc
CXX = g++
CC_FLAGS = -O3
LOGS?=1

#Three archs
# x86 arm xeon
ARCH?=x86

ifeq ($(ARCH), xeon) 
CC_FLAGS+= -qopenmp -DBOFFSET=12 icc -mmic -DMIC_NATIVE

else ifeq ($(ARCH), arm)
CC_FLAGS+= -fopenmp -DBOFFSET=12 -static -mfloat-abi=hard -mfpu=auto

else ifeq ($(ARCH), x86)
CC_FLAGS+= -fopenmp -DBOFFSET=12

endif

ifeq ($(LOGS), 1)
INCLUDE+= -I../../include
LIBRARY+= -L../../include -lLogHelper
endif

all: dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices

dgemm_gen: dgemm_gen.c ../../include/log_helper.c
	$(CC)  dgemm_gen.c -o dgemm_gen $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

dgemm_check: dgemm.c ../../include/log_helper.c
	$(CC)  dgemm.c -DLOGS -o dgemm_check $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

dgemm_err_inj: dgemm.c  ../../include/log_helper.c
	$(CC) dgemm.c -DLOGS -DERR_INJ -o dgemm_err_inj $(CC_FLAGS)  $(INCLUDE) $(LIBRARY)

dgemm_timing: dgemm.c ../../include/log_helper.c
	$(CC)  dgemm.c -DTIMING -o dgemm_timing $(CC_FLAGS) $(INCLUDE) $(LIBRARY)

genMatrices: generateMatrices.cpp
	$(CXX) generateMatrices.cpp -o genMatrices

clean:
	rm -f dgemm_gen dgemm_check dgemm_err_inj dgemm_timing genMatrices
