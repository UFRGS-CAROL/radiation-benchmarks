CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
PRECISION?=single
LOGS?=1
USE_OMP?=0
BUILDPROFILER=0
FORJETSON=0
ITERATIONS=5

UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

# This is what I use, uncomment if you know your arch and want to specify
# Kepler, Volta
SMS= 35 53 70 86

$(foreach sm,$(SMS), $(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))


NVCC_FLAGS= $(ARCH) -O3 -std=c++11 -ccbin $(CXX)
INCLUDE= -I$(CUDAPATH)/include
LIBRARY= -L$(CUDAPATH)/lib64/
CXXFLAGS= -I$(CUDAPATH)/include -L$(CUDAPATH)/lib64/ -lcudart  -std=c++11

TARGET=cuda_lava_$(LOWERCASED_PRECISION)
RUN_SOURCES=./cuda_lava.cu

LOGHELPER_INC=../../include/,../common
LOGHELPER_LIB=../../include/,../common/lib



ifeq ($(LOGS), 1)
INCLUDE+= -I$(LOGHELPER_INC)
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
	ifeq ($(FORJETSON), 1)
	NVCC_FLAGS += -DBUILDPROFILER -DFORJETSON=1 -lJTX2Inst -li2c
	else
	NVCC_FLAGS += -DBUILDPROFILER -L$(CUDAPATH)/lib64/stubs -lNVMLWrapper -lnvidia-ml
	endif
endif 

NVCC_FLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

DATA_DIR=.
SIZE=23
STREAMS=1
DEF_CHARGES_INPUT=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_charges_$(SIZE)
DEF_DISTANCES_INPUT=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_distances_$(SIZE)
DEF_GOLD=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_gold_$(SIZE)

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET) 

clean:
	rm -rf ./$(TARGET) $(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD)

generate:
	./$(TARGET) -boxes=$(SIZE) -generate -streams=$(STREAMS) -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: 
	./$(TARGET) -boxes=$(SIZE) -streams=$(STREAMS) -iterations=$(ITERATIONS) -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
