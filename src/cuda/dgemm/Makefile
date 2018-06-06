CC=g++
NVCC=/usr/local/cuda/bin/nvcc
LOGS?=1
USE_OMP?=1


ARCH=  -gencode arch=compute_35,code=[sm_35,compute_35] 
ARCH+= -gencode arch=compute_50,code=[sm_50,compute_50] 
ARCH+= -gencode arch=compute_52,code=[sm_52,compute_52] 
ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] 
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62]
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]


NVCC_FLAGS= $(ARCH) -O3  -lcublas -std=c++11
INCLUDE= -I/usr/local/cuda/include #/ -I/usr/local/cuda/samples/common/inc/
LIBRARY= -L/usr/local/cuda/lib64/

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

ifeq ($(LOGS), 1)
INCLUDE+= -I$(LOGHELPER_INC)
NVCC_FLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB)
endif


ifeq ($(USE_OMP), 1)
NVCC_FLAGS+= -DUSE_OMP -Xcompiler " -fopenmp -O3"
endif

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

BIN=.


RUN_SOURCES=$(BIN)/cudaDGEMM.cu
GEN_SOURCES=$(BIN)/generateMatricesDouble.cu
RUN_EXEC=$(BIN)/cudaDGEMM
GEN_EXEC=$(BIN)/generateMatricesDouble
DEF_SIZE=8192
DEF_A_INPUT=./dgemm_A_8192.matrix
DEF_B_INPUT=./dgemm_B_8192.matrix
DEF_GOLD=./dgemm_GOLD_$(DEF_TEST_SIZE).matrix
MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./dgemm*.matrix


SRC = $(wildcard *.cu)
TARGET = $(SRC:%.cu=$(BIN)/%)

all: $(TARGET)

$(BIN)/% : %.cu
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(RUN_EXEC) $(GEN_EXEC) $(MATRIX_NAMES)

generate: all
	$(GEN_EXEC) -size=$(DEF_SIZE)  -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -use_tensors=1

test: generate
	$(RUN_EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10  -use_tensors=1 -verbose
