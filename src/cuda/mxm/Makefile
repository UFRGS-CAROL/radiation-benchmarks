CC=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
PRECISION?=single
FORJETSON=0
USE_OMP?=0
DEF_SIZE=4096
#8192

LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(LOWERCASED_PRECISION),half)
ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	#Tegra X1
ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif

NVCC_FLAGS= $(ARCH) -Xptxas -v

NVIDIA_INC=$(CUDAPATH)/include/,./
NVIDIA_LIB=$(CUDAPATH)/lib64/
LOGHELPER_INC=../../include/,../common
LOGHELPER_LIB=../../include/,../common/lib


ifeq ($(FORJETSON), 1)
NVCC_FLAGS += -DFORJETSON=1 -lJTX2Inst -li2c
else
NVCC_FLAGS += -L$(CUDAPATH)/lib64/stubs 

endif

ifeq ($(USE_OMP), 1)
NVCC_FLAGS+= -DUSE_OMP -Xcompiler " -fopenmp"
endif

SOURCES=./cuda_mxm.cu
EXEC_RADIX=./cuda_mxm_
EXEC=$(EXEC_RADIX)$(LOWERCASED_PRECISION)

DEF_A_INPUT=./mxm_$(LOWERCASED_PRECISION)_A_8192.matrix
DEF_B_INPUT=./mxm_$(LOWERCASED_PRECISION)_B_8192.matrix
DEF_GOLD=./mxm_$(LOWERCASED_PRECISION)_GOLD_$(DEF_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./mxm*.matrix

all: $(EXEC)

check-benchtype:
	echo "syntax: make PRECISION=<double|half|single>"
	test $(PRECISION)

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -Dtest_precision_$(LOWERCASED_PRECISION) -O3 \
			-std=c++11 $(NVCC_FLAGS) -Xcompiler " -O3" $(LOGS) $(SOURCES) -o $(EXEC) -lLogHelper

clean:
	rm -rf $(EXEC_RADIX)* $(MATRIX_NAMES)

generate:
	$(EXEC) -generate -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: 
	$(EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=1 -verbose
