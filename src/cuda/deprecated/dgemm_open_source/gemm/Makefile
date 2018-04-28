# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= $(SCC_CUDA_DIR)/install
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++
PGCC            ?= /usr/local/apps/pgi-13.5/linux86-64/13.5/bin/pgcc

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LDFLAGS   ?=
EXTRA_CCFLAGS   ?=

# CUDA code generation flags
GENCODE_FLAGS    := -gencode arch=compute_20,code=sm_20

      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64

      NVCCFLAGS := -m64

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I. 
LDFLAGS       += -lcublas -lblas

# PGCC code generation flags
PGCCFLAGS     := -acc -Minfo

# Target rules
all: build

build: cuGEMM accGEMM cuGEMM_test

cuGEMM_test.o: cuGEMM_test.cu
	$(NVCC) $(NVCCFLAGS) -I$(CUDA_INC_PATH) $(GENCODE_FLAGS) $(INCLUDES) -O3 -o $@ -c $<

cuGEMM_test: cuGEMM_test.o
	$(GCC) $(CCFLAGS) -O3 -o $@ $+ $(LDFLAGS) 

cuGEMM.o: cuGEMM.cpp
	$(GCC) $(INCLUDES) -O3 -o $@ -c $<

cuGEMM: cuGEMM.o
	$(GCC) $(CCFLAGS) -O3 -o $@ $+ $(LDFLAGS) 

accGEMM.o: accGEMM.c
	$(PGCC) $(INCLUDES) -O3 -o $@ -c $<

accGEMM: accGEMM.o
	$(PGCC) $(PGCCFLAGS) -o $@ $+ 

run: build
	./cuGEMM ./accGEMM

clean:
	rm -f cuGEMM_test.o cuGEMM_test cuGEMM.o cuGEMM accGEMM.o accGEMM

