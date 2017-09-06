CC=g++
NVCC=nvcc

NVCC_FLAGS=-arch=sm_53

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=cudaHGEMM.cu
SOURCES_GEN=generateMatricesHalf.cu
RUN_EXEC=cudaHGEMM
GEN_EXEC=generateMatricesHalf

MATRIX_NAMES=hgemm_*


all: gemm gen

gemm: cudaHGEMM.cu
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -O3 -lLogHelper -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" $(LOGS) cudaHGEMM.cu -o $(RUN_EXEC)

gen: generateMatricesHalf.cu
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) $(NVCC_FLAGS) -lcublas -O3 generateMatricesHalf.cu -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)
