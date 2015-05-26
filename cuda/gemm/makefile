CC=g++
NVCC=nvcc

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=cudaGEMM.cu
SOURCES_GEN=generateMatrices.cu
RUN_EXEC=cudaGEMM
GEN_EXEC=generateMatrices

MATRIX_NAMES=Double_*.matrix 


all: gemm gen

gemm: cudaGEMM.cu
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) cudaGEMM.cu -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -lLogHelper -lcublas -o $(RUN_EXEC)

gen: generateMatrices.cu
	$(NVCC) generateMatrices.cu -I$(NVIDIA_INC) -L$(NVIDIA_LIB) -lcublas -o $(GEN_EXEC)
	
clean:
	rm $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

