CC=g++
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
ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]


NVCC_FLAGS= $(ARCH) -O3  -lcublas -std=c++11 -lSafeMemory
INCLUDE= -I/usr/local/cuda/include #/ -I/usr/local/cuda/samples/common/inc/
LIBRARY= -L/usr/local/cuda/lib64/ -L../../include/safe_memory/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11

TARGET=hotspot

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

RUN_SOURCES=./hotspot.cu
SAFEMEMORYDIR=../dgemm

NVCC_FLAGS+= -I$(SAFEMEMORYDIR)/

all: $(GEN_EXEC) $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS)  $(RUN_SOURCES) -o $(TARGET)
	
#safe_memory.o: $(SAFEMEMORYDIR)/safe_memory.cpp $(SAFEMEMORYDIR)/safe_memory.h
#	$(CXX) $(COMMON) $(CFLAGS) -c $(SAFEMEMORYDIR)/safe_memory.cpp -o $@ $(LDFLAGS) $(CFLAGS) $(CXXFLAGS) 

clean:
	rm -f hotspot gold_*
