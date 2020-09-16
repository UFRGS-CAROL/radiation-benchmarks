CUDAPATH=/usr/local/cuda
CC=gcc
CXX=g++
NVCC=$(CUDAPATH)/bin/nvcc
PRECISION?=single
DEF_SIZE?=1024
LOGS?=1
USE_OMP?=0

BUILDPROFILER=0

# NVCC optmizations flags
# To use for different flags
NVCCOPTFLAGS?=

UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

TEST_SIM_TIME=10000
DATA_DIR=../../../data/hotspot
DEF_INPUT_POWER=$(DATA_DIR)/power_$(DEF_SIZE)
DEF_INPUT_TEMP=$(DATA_DIR)/temp_$(DEF_SIZE)
DEF_GOLD=$(DATA_DIR)/gold_temp_$(LOWERCASED_PRECISION)_$(DEF_SIZE)_$(TEST_SIM_TIME)


ifeq ($(UPPERCASED_PRECISION),HALF)
ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else

ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
#ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	# Tegra X1
#ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
#ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100

# check for cuda 7.0 or lower
ifneq (, $(findstring cuda-7.0, $(CUDAPATH)))
else
# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]
# Tegra X2
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]
# Titan V | Tesla V100
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]
# Xavier
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]
endif

endif

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

NVCC_FLAGS= $(ARCH) -std=c++11 -Xptxas -v $(NVCCOPTFLAGS)
INCLUDE= -I$(CUDAPATH)/include -I../common/include  -I$(LOGHELPER_INC)
LIBRARY= -L$(CUDAPATH)/lib64/
CXXFLAGS= -I$(CUDAPATH)/include -L$(CUDAPATH)/lib64/ -lcudart  -std=c++11

TARGET=cuda_hotspot_$(LOWERCASED_PRECISION)
RUN_SOURCES=./cuda_hotspot.cu


ifeq ($(LOGS), 1)
#INCLUDE+= -I$(LOGHELPER_INC)
NVCC_FLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB) 
CXXFLAGS+= -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -DLOGS -lLogHelper 
endif

ifeq ($(USE_OMP), 1)
NVCC_FLAGS+= -DUSE_OMP -Xcompiler " -fopenmp -O3"
endif

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

ifeq ($(BUILDPROFILER), 1)
NVCC_FLAGS+= -DBUILDPROFILER
LIBRARY+= -L../common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif


NVCC_FLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

NVCC_FLAGS+= -DNVCCOPTFLAGS="$(strip $(patsubst %-Xptxas,%,$(NVCCOPTFLAGS)))"  

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) -ccbin $(CXX) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET)

clean:
	rm -rf $(TARGET) $(DEF_GOLD)

generate: $(TARGET)
	./$(TARGET) -generate -size=$(DEF_SIZE) -input_power=$(DEF_INPUT_POWER) \
				-input_temp=$(DEF_INPUT_TEMP) -gold_temp=$(DEF_GOLD) \
				-sim_time=$(TEST_SIM_TIME) -verbose -iterations=1 -streams=1

test: $(TARGET) generate
	./$(TARGET) -size=$(DEF_SIZE) -input_power=$(DEF_INPUT_POWER) \
				-input_temp=$(DEF_INPUT_TEMP) -gold_temp=$(DEF_GOLD) \
				-sim_time=$(TEST_SIM_TIME) -iterations=10 -verbose
				
