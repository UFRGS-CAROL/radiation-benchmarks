OPS?=100
OPSNUM=1
LOGS=1
DEBUG?=0
CHECKBLOCK=0
ZERO_FLOAT=2.2e-07
MICRO=mul
MINRAND=0
MAXRAND=100
BENCH=mul

CC=g++
NVCC=nvcc

ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -Xptxas -v

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

ifeq ($(DEBUG), 1)
OPS=1000
NVCC_FLAGS+= -g -G
endif


ifeq ($(LOGS), 1)
NVCC_FLAGS+= -DLOGS
FLAGS=-I$(LOGHELPER_INC) -L$(LOGHELPER_INC) 
endif

SOURCES=./cuda_micro.cu
EXEC=./cuda_micro_mp_hardening

NVCC_FLAGS+= -DCHECKBLOCK=$(CHECKBLOCK) -DZERO_FLOAT=$(ZERO_FLOAT)


FLAGS+=-I$(NVIDIA_INC) -I../common -L$(NVIDIA_LIB) -DOPS=$(OPS) 
FLAGS+= -O3 -lLogHelper -std=c++11  $(NVCC_FLAGS)


.PHONY: all test clean generate

all: $(EXEC)

$(EXEC): clean $(SOURCES)
	$(NVCC) $(FLAGS) -Xcompiler "-fopenmp -O3" $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC)

generate:
	$(EXEC) --precision double --redundancy dmrmixed \
		--inst $(MICRO) --generate --opnum $(OPSNUM) --nonconstant \
		--minrand $(MINRAND) --maxrand $(MAXRAND) \
		--gold ./gold_mul.data --verbose
		
test:
	$(EXEC) --precision double --redundancy dmrmixed \
		--inst $(MICRO) --opnum $(OPSNUM) --nonconstant \
		--minrand $(MINRAND) --maxrand $(MAXRAND) \
		--gold ./gold_mul.data --verbose --iterations 10
