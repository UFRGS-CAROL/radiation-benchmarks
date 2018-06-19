CC=g++
NVCC=/usr/local/cuda/bin/nvcc
PRECISION?=single
LOGS?=1
USE_OMP?=1

UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(UPPERCASED_PRECISION),HALF)
ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	# Tegra X1
ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif


NVCC_FLAGS= $(ARCH) -O3 -std=c++11
INCLUDE= -I/usr/local/cuda/include
LIBRARY= -L/usr/local/cuda/lib64/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11

TARGET=cuda_lava_$(LOWERCASED_PRECISION)
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

NVCC_FLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

DATA_DIR=.
DEF_SIZE=40
DEF_CHARGES_INPUT=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_charges_$(DEF_SIZE)
DEF_DISTANCES_INPUT=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_distances_$(DEF_SIZE)
DEF_GOLD=$(DATA_DIR)/lava_$(LOWERCASED_PRECISION)_gold_$(DEF_SIZE)

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET)

clean:
	rm -rf ./$(TARGET) $(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD)

generate: $(TARGET)
	./$(TARGET) -boxes=$(DEF_SIZE) -generate -streams=1 -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	./$(TARGET) -boxes=$(DEF_SIZE) -streams=1 -iterations=10 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
