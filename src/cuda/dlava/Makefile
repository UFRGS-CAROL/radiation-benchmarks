CXX=g++
NVCC=/usr/local/cuda/bin/nvcc
LOGS?=1
USE_OMP?=1
SAFE_MALLOC?=1

ARCH=  -gencode arch=compute_35,code=[sm_35,compute_35] # K40
ARCH+= -gencode arch=compute_53,code=[sm_53,compute_53] # Tegra X1
ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] # Tesla P100
ARCH+= -gencode arch=compute_61,code=[sm_61,compute_61] # Titan X | Titan Xp
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62] # Tegra X2 
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] # Titan V | Tesla V100


NVCC_FLAGS= $(ARCH) -O3  -lcublas -std=c++11  -lSafeMemory
INCLUDE= -I/usr/local/cuda/include #/ -I/usr/local/cuda/samples/common/inc/
LIBRARY= -L/usr/local/cuda/lib64/ -L../../include/safe_memory
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
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

ifeq ($(SAFE_MALLOC), 1)
NVCC_FLAGS+= -DSAFE_MALLOC
endif

RUN_SOURCES=./dlava.cu
RUN_EXEC=./dlava

DEF_SIZE=12

DEF_CHARGES_INPUT=./dlava_charges_$(DEF_SIZE)
DEF_DISTANCES_INPUT=./dlava_distances_$(DEF_SIZE)
DEF_GOLD=./dlava_gold_$(DEF_SIZE)

SAFEMEMORYDIR=../dgemm
NVCC_FLAGS+= -I$(SAFEMEMORYDIR)/
NVCC_FLAGS+= -I$(SAFEMEMORYDIR)/
INPUTS_NAMES=$(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD) ./dlava_charges_* ./dlava_distances_* ./dlava_gold_*

ALL: $(RUN_EXEC)

$(RUN_EXEC):  $(RUN_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE) $(LIBRARY) $(RUN_SOURCES)  -o $(RUN_EXEC)

#safe_memory.o: $(SAFEMEMORYDIR)/safe_memory.cpp $(SAFEMEMORYDIR)/safe_memory.h
#	$(CXX) $(COMMON) $(CFLAGS) -c $(SAFEMEMORYDIR)/safe_memory.cpp -o $@ $(LDFLAGS) $(CFLAGS) $(CXXFLAGS) 


$(GEN_EXEC): $(GEN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(GEN_SOURCES) -o $(GEN_EXEC)
	

#safe_memory.o: $(SAFEMEMORYDIR)/safe_memory.cpp $(SAFEMEMORYDIR)/safe_memory.h
#	$(CXX) $(COMMON) $(CFLAGS) -c $(SAFEMEMORYDIR)/safe_memory.cpp -o $@ $(LDFLAGS) $(CFLAGS) $(CXXFLAGS) 

clean:
	rm -f $(RUN_EXEC) $(INPUTS_NAMES) *.o

generate: $(RUN_EXEC)
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -generate -streams=1 -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -streams=1 -iterations=10 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
