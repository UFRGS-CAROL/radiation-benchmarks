CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_30,code=[sm_30,compute_30] \
      -gencode arch=compute_35,code=[sm_35,compute_35] \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_SOURCES=./cuda_trip_dmxm.cu
GEN_SOURCES=./generateMatricesDouble.cu
RUN_EXEC=./cuda_trip_dmxm
GEN_EXEC=./generateMatricesDouble

DEF_SIZE=1024

DEF_A_INPUT=./dmxm_A_8192.matrix
DEF_B_INPUT=./dmxm_B_8192.matrix
DEF_GOLD=./dmxm_GOLD_$(DEF_TEST_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./dmxm*.matrix


all: $(RUN_EXEC) $(GEN_EXEC)

$(RUN_EXEC): $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -O3 -lLogHelper -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" $(LOGS) $(RUN_SOURCES) -o $(RUN_EXEC)

$(GEN_EXEC): $(GEN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC) -L$(NVIDIA_LIB) $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" -lcublas -O3 $(GEN_SOURCES) -o $(GEN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

generate: $(GEN_EXEC)
	$(GEN_EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: $(RUN_EXEC) $(GEN_EXEC) generate
	$(RUN_EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose
