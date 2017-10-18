CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH) #-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=cudaHGEMM.cu
SOURCES_GEN=generateMatricesHalf.cu
RUN_EXEC=cudaHGEMM
GEN_EXEC=generateMatricesHalf

MATRIX_NAMES=hgemm*


all: gemm gen

gemm: cudaHGEMM.cu
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -O3 -lLogHelper -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" $(LOGS) cudaHGEMM.cu -o $(RUN_EXEC)

gen: generateMatricesHalf.cu
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) $(NVCC_FLAGS) -lcublas -O3 generateMatricesHalf.cu -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)
