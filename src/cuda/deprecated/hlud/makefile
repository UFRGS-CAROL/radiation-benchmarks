CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -std=c++11 #-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_SOURCES=./cudaHLUD.cu
RUN_EXEC=./cudaHLUD

DEF_SIZE=1024

DEF_INPUT=./HLUD_input_8192.matrix
DEF_GOLD=./HLUD_gold_$(DEF_SIZE).matrix

MATRIX_NAMES=$(DEF_INPUT) $(DEF_GOLD) ./HLUD*.matrix


all: $(RUN_EXEC)

$(RUN_EXEC): $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -O3 -lLogHelper -lcublas $(NVCC_FLAGS) -Xcompiler " -fopenmp -O3" $(LOGS) $(RUN_SOURCES) -o $(RUN_EXEC)

clean:
	rm -rf $(RUN_EXEC) $(MATRIX_NAMES)

generate: $(GEN_EXEC)
	$(RUN_EXEC) -size=$(DEF_SIZE) -generate -input=$(DEF_INPUT) -gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	$(RUN_EXEC) -size=$(DEF_SIZE) -input=$(DEF_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose
