OPS?=1000000000
DEBUG?=0

CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -Xptxas -v

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

SOURCES=./cuda_micro.cu
EXEC_RADIX=./cuda_micro-
EXEC=$(EXEC_RADIX)_mp_hardening

FLAGS=-I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DOPS=$(OPS) -DLOGS  -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS)

all: $(EXEC)

$(EXEC): clean $(SOURCES)
	$(NVCC) $(FLAGS) -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)*

test: $(EXEC)
	$(EXEC) -verbose -iterations=10
