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
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -lLogHelperCuda -lcublas $(NVCCFLAGS) cudaGEMM.cu -o $(RUN_EXEC)

gen: generateMatrices.cu
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) -lcublas generateMatrices.cu -o $(GEN_EXEC)
	
clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

