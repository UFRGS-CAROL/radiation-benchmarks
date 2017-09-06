CC=g++
NVCC=nvcc

NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=cudaDGEMM.cu
SOURCES_GEN=generateMatricesDouble.cu
RUN_EXEC=cudaDGEMM
GEN_EXEC=generateMatricesDouble

MATRIX_NAMES=dgemm_*


all: gemm gen

gemm: cudaDGEMM.cu
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -lLogHelper -O3 -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" -DLOGS cudaDGEMM.cu -o $(RUN_EXEC)

gen: generateMatricesDouble.cu
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) $(NVCC_FLAGS) -O3 -lcublas generateMatricesDouble.cu -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)


generate:
	./generateMatricesDouble -size=1024 -input_a=dgemmA_8192.matrix -input_b=dgemmB_8192.matrix -gold=dgemmGOLD_1024.matrix

test:
	./cudaDGEMM -size=1024 -input_a=dgemmA_8192.matrix -input_b=dgemmB_8192.matrix -gold=dgemmGOLD_1024.matrix -iterations=10 -verbose

