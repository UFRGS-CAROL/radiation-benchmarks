CC=g++
NVCC=nvcc

NVCC_FLAGS=-arch=sm_53

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=cudaGEMM.cu
SOURCES_GEN=generateMatrices.cu
RUN_EXEC=cudaGEMM
GEN_EXEC=generateMatrices

MATRIX_NAMES=hgemm_*


all: gemm gen

gemm: cudaGEMM.cu
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -lLogHelper -lcublas $(NVCCFLAGS) $(LOGS) cudaGEMM.cu -o $(RUN_EXEC)

gen: generateMatrices.cu
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) -lcublas generateMatrices.cu -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)
