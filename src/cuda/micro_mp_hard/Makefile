OPS?=1000
LOGS=1
DEBUG?=0
MICRO=mul
DMR=none
CHECK_BLOCK=$(OPS)

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

SOURCES=./src/main.cu
EXEC=./cuda_micro_mp_hardening

FLAGS+=-I$(NVIDIA_INC) -I../common/include -L$(NVIDIA_LIB) -DOPS=$(OPS) 
FLAGS+= -O3 -lLogHelper -std=c++11  $(NVCC_FLAGS)


.PHONY: all test clean generate

all: $(EXEC) generate_input

$(EXEC): clean $(SOURCES)
	$(NVCC) $(FLAGS) -Xcompiler "-fopenmp -O3" $(SOURCES) -o $(EXEC)

generate_input: clean
	$(CC) -O3 -std=c++11 -o generate_input generate_input.cpp

clean:
	rm -rf $(EXEC) generate_input

generate:
	$(EXEC) --precision double  \
			--redundancy $(DMR) \
			--inst $(MICRO)     \
			--generate \
			--gold ./gold_$(MICRO)_$(DMR).data \
			--opnum $(CHECK_BLOCK) \
			--verbose
		
test:
	$(EXEC) --precision double  \
			--redundancy $(DMR) \
			--inst $(MICRO)     \
			--gold ./gold_$(MICRO)_$(DMR).data \
			--opnum $(CHECK_BLOCK) \ 
			--verbose \
			--iterations 1
