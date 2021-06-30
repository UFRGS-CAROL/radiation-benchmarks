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

# Kepler, Volta
SMS= 35

ifeq ($(UPPERCASED_PRECISION),HALF)
SMS+= 53 60
else

# check for cuda 7.0 or lower
ifneq (, $(findstring cuda-7.0, $(CUDAPATH)))
else
SMS+= 70
endif
endif

$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))


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

# TODO: verify this code
NVCC_FLAGS+= -DNVCCOPTFLAGS="$(strip $(patsubst %-Xptxas,%,$(NVCCOPTFLAGS)))"  
# Host must be always at maximum opt
NVCC_FLAGS+= --optimize 3

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
				
