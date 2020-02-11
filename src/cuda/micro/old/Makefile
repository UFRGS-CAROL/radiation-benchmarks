OPS?=1000000000
PRECISION?=single
DEBUG?=0
TYPE=fma
BUILDPROFILER=0

CC=g++
NVCC=/usr/local/cuda/bin/nvcc

LOWERCASED_TYPE=$(shell echo $(TYPE) | tr A-Z a-z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(LOWERCASED_PRECISION),half)
ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]  	# Tesla K40
ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]  	# Tegra X1
ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif


NVCC_FLAGS= $(ARCH) -Xptxas -v

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/,../common
LOGHELPER_LIB=../../include/,../common/lib

ifeq ($(BUILDPROFILER), 1)

NVCC_FLAGS+= -DBUILDPROFILER=1

ifeq ($(FORJETSON), 1)
NVCC_FLAGS += -DFORJETSON=1 -lJTX2Inst -li2c
else
NVCC_FLAGS += -L/usr/local/cuda/lib64/lib64/stubs -lnvidia-ml  -lNVMLWrapper

endif

endif

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

SOURCES=./cuda_micro.cu
EXEC_RADIX=./cuda_micro-
EXEC=$(EXEC_RADIX)$(LOWERCASED_TYPE)_$(LOWERCASED_PRECISION)

all: $(EXEC)

check-benchtype:
	echo "syntax: make TYPE=<fma|add|mul> PRECISION=<double|half|single>"
	test $(TYPE)
	test $(PRECISION)

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DOPS=$(OPS) -DLOGS -Dtest_precision_$(LOWERCASED_PRECISION) -Dtest_type_$(LOWERCASED_TYPE) -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS) -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)*

test: $(EXEC)
	$(EXEC) -verbose -iterations=10
