CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
LOGS?=1
USE_OMP?=1
SAFE_MALLOC?=1

ARCH=  -gencode arch=compute_35,code=[sm_35,compute_35] 
ARCH+= -gencode arch=compute_50,code=[sm_50,compute_50] 
ARCH+= -gencode arch=compute_52,code=[sm_52,compute_52] 
ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] 
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62]
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]


NVCC_FLAGS= $(ARCH) -O3  -lcublas -std=c++11
INCLUDE= -I/usr/local/cuda/include #/ -I/usr/local/cuda/samples/common/inc/
LIBRARY= -L/usr/local/cuda/lib64/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart


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

ifeq ($(SAFE_MALLOC), 1)
NVCC_FLAGS+= -DSAFE_MALLOC
endif

BIN=.

SAFEMEMORYDIR=../dgemm
OBJS=$(SAFEMEMORYDIR)/safe_memory.o
RUN_SOURCES=$(BIN)/cudaHGEMM.cu
GEN_SOURCES=$(BIN)/generateMatricesHalf.cu
TARGET=$(BIN)/cudaHGEMM
GEN_EXEC=$(BIN)/generateMatricesHalf
DEF_SIZE=8192
DEF_A_INPUT=./hgemm_A_8192.matrix
DEF_B_INPUT=./hgemm_B_8192.matrix
DEF_GOLD=./hgemm_GOLD_$(DEF_TEST_SIZE).matrix
MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./hgemm*.matrix

NVCC_FLAGS+= -I$(SAFEMEMORYDIR)/

all: $(GEN_EXEC) $(TARGET)

$(TARGET): safe_memory.o $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) safe_memory.o $(RUN_SOURCES) -o $(TARGET)
	
$(GEN_EXEC): $(GEN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(GEN_SOURCES) -o $(GEN_EXEC)
	

safe_memory.o: $(SAFEMEMORYDIR)/safe_memory.cpp $(SAFEMEMORYDIR)/safe_memory.h
	$(CXX) $(COMMON) $(CFLAGS) -c $(SAFEMEMORYDIR)/safe_memory.cpp -o $@ $(LDFLAGS) $(CFLAGS) $(CXXFLAGS) 

clean:
	rm -rf $(TARGET) $(GEN_EXEC) $(MATRIX_NAMES) *.o

generate: all
	$(GEN_EXEC) -size=$(DEF_SIZE)  -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -use_tensors=1

test: generate
	$(TARGET) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10  -use_tensors=1 -verbose
