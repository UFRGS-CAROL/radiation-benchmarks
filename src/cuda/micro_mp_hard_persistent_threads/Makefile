OPS?=1000000
LOGS=1
DEBUG?=0
FORJETSON=0

CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Titan V | Tesla V100
ARCH+= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -Xptxas -v

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/,../common/lib
LOGHELPER_LIB=../../include/,../common/lib


ifeq ($(FORJETSON), 1)
NVCC_FLAGS += -DFORJETSON=1 -lJTX2Inst -li2c
else
NVCC_FLAGS += -L$(NVIDIA_LIB)/stubs -lnvidia-ml  -lNVMLWrapper
endif


ifeq ($(DEBUG), 1)
OPS=1000
NVCC_FLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
NVCC_FLAGS+= -DLOGS
FLAGS=-I$(LOGHELPER_INC) -L$(LOGHELPER_INC) 
endif

SOURCES=./cuda_micro.cu
EXEC=./cuda_micro_persistent_threads

FLAGS+=-I$(NVIDIA_INC) -I../common -L$(NVIDIA_LIB) -DOPS=$(OPS)
FLAGS+= -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS)

all: $(EXEC)

$(EXEC): clean $(SOURCES)
	$(NVCC) $(FLAGS) -Xcompiler "-fopenmp -O3" $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC)

test:
	$(EXEC) --verbose --iterations 10 --precision single --redundancy none --inst fma
