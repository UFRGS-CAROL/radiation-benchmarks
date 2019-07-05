CC=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
PRECISION?=single
BLOCK_SIZE=32

LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

FORJETSON=1

#ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
#ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH)

ifneq ($(BLOCK_SIZE), 32)
NVCC_FLAGS += -DBLOCK_SIZE=16
endif

ifeq ($(FORJETSON), 1)
NVCC_FLAGS += -DFORJETSON=1 -lJTX2Inst -li2c
else
NVCC_FLAGS += -L$(CUDAPATH)/lib64/stubs -lnvidia-ml  -lNVMLWrapper

endif


NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/,../common/lib
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

PERSISTENT_INC=../common/

SOURCES=./cuda_trip_mxm.cu
EXEC_RADIX=./cuda_mxm_persistent_threads_
EXEC=$(EXEC_RADIX)$(LOWERCASED_PRECISION)

DEF_SIZE=64

DEF_A_INPUT=./mxm_$(LOWERCASED_PRECISION)_A_$(DEF_SIZE).matrix
DEF_B_INPUT=./mxm_$(LOWERCASED_PRECISION)_B_$(DEF_SIZE).matrix
DEF_GOLD=./mxm_$(LOWERCASED_PRECISION)_GOLD_$(DEF_TEST_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./mxm*.matrix

all: $(EXEC)

check-benchtype:
	echo "syntax: make PRECISION=<double|half|single>"
	test $(PRECISION)

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC),$(PERSISTENT_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB)  \
		 -DLOGS -Dtest_precision_$(LOWERCASED_PRECISION) -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS) \
		 -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)* $(MATRIX_NAMES)

generate: $(EXEC)
	$(EXEC) -generate -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: $(EXEC) 
	$(EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose
