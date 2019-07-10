CC=g++
NVCC=/usr/local/cuda/bin/nvcc
PRECISION?=single
LOGS?=1
USE_OMP?=1

FORJETSON=0


UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
#ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -O3 -std=c++11 -L../common/lib
INCLUDE= -I/usr/local/cuda/include -I../common
LIBRARY= -L/usr/local/cuda/lib64/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/  -lcudart  -std=c++11 

TARGET=cuda_lava_persistent_threads_$(LOWERCASED_PRECISION)
RUN_SOURCES=./cuda_lava.cu

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

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


ifeq ($(FORJETSON), 1)
NVCC_FLAGS += -DFORJETSON=1 -lJTX2Inst -li2c
else
NVCC_FLAGS += -L/usr/local/cuda/lib64/stubs -lnvidia-ml  -lNVMLWrapper
endif


NVCC_FLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

DATA_DIR=.
DEF_SIZE=2
DEF_CHARGES_INPUT=$(DATA_DIR)/lava_charges_$(LOWERCASED_PRECISION)_$(DEF_SIZE)
DEF_DISTANCES_INPUT=$(DATA_DIR)/lava_distances_$(LOWERCASED_PRECISION)_$(DEF_SIZE)
DEF_GOLD=$(DATA_DIR)/lava_gold_$(LOWERCASED_PRECISION)_$(DEF_SIZE)

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET)

clean:
	rm -rf ./$(TARGET) $(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD)

generate: $(TARGET)
	./$(TARGET) -boxes=$(DEF_SIZE) -generate -streams=1 -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	./$(TARGET) -boxes=$(DEF_SIZE) -streams=1 -iterations=10 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
