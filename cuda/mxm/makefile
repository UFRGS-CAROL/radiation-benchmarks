CC=g++
NVCC=nvcc

NVIDIA_INC=/usr/local/cuda/include
NVIDIA_LIB=/usr/local/cuda/lib64
LOGHELPER_INC=../../include
LOGHELPER_LIB=../../include

SOURCES=cudaMxM.cu
SOURCES_GEN=generateMatrices.cu
RUN_EXEC=cudaMxM
GEN_EXEC=generateMatrices

MATRIX_NAMES=Double_*.matrix 


all: mxm gen

mxm: cudaMxM.cu
	$(NVCC) -I$(LOGHELPER_INC) cudaMxM.cu -L$(LOGHELPER_LIB) -lLogHelperCuda $(NVCCFLAGS)  -o $(RUN_EXEC)

gen: generateMatrices.cu
	$(NVCC) generateMatrices.cu -I$(NVIDIA_INC) -o $(GEN_EXEC)
	
clean:
	rm $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

