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
LIBRARY= -L/usr/local/cuda/lib64/ -L../../include/safe_memory
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11

TARGET=cuda_trip_hotspot

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

RUN_SOURCES=./cuda_trip_hotspot.cu
SAFEMEMORYDIR=../dgemm

NVCC_FLAGS+= -I$(SAFEMEMORYDIR)/

all: $(GEN_EXEC) $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET)
	
test: $(TARGET)
	./cuda_trip_hotspot -size=1024 -sim_time=1024 -temp_file=../../../data/hotspot/temp_1024 -power_file=../../../data/hotspot/power_1024 -gold_file=gold_test -iterations=10 -streams=1 -verbose
	
	
generate:
	./cuda_trip_hotspot -generate -size=1024 -sim_time=1024 -temp_file=../../../data/hotspot/temp_1024 -power_file=../../../data/hotspot/power_1024 -gold_file=gold_test -iterations=1	 -streams=1 -verbose

	

clean:
	rm -f $(TARGET) gold_*
