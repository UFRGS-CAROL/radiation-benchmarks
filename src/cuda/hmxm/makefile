CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_SOURCES=./cudaHMxM.cu
GEN_SOURCES=./generateMatricesHalf.cu
RUN_EXEC=./hmxm
GEN_EXEC=./generateMatricesHalf

DEF_SIZE=1024

DEF_A_INPUT=./hmxm_A_8192.matrix
DEF_B_INPUT=./hmxm_B_8192.matrix
DEF_GOLD=./hmxm_GOLD_$(DEF_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD)


all: $(RUN_EXEC) $(GEN_EXEC)

$(RUN_EXEC): $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -O3 -lLogHelper -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" $(LOGS) $(RUN_SOURCES) -o $(RUN_EXEC)

$(GEN_EXEC): $(GEN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" -lcublas -O3 $(GEN_SOURCES) -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

generate: $(GEN_EXEC)
	$(GEN_EXEC) -size=$(DEF_SIZE) -host_check -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: $(RUN_EXEC) $(GEN_EXEC) generate
	$(RUN_EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose