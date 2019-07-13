LOGS=1
DEBUG?=0

CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

#ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
ARCH+= -gencode arch=compute_35,code=[sm_35,compute_35]	# Titan V | Tesla V100
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]	# Jetson Xavier

NVCCFLAGS= $(ARCH) -Xptxas -v -std=c++11
CXXFLAGS= -std=c++11 -fopenmp -O3
INCLUDE= -I$(CUDAPATH)/include/ -I../../include/ -I../common/include/ -I$(CUDAPATH)/samples/common/inc/
LIBRARY= -L$(CUDAPATH)/lib64/  -L../../include/ -L../common/lib -L$(CUDAPATH)/lib64/stubs

OBJDIR=./obj/

EXEC=cuda_batched_mxm


ifeq ($(DEBUG), 1)
NVCCFLAGS+= -g -G
CXXFLAGS+= -g
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
LIBRARY+= -lLogHelper
endif

COMMON= $(INCLUDE)
LDFLAGS= $(LIBRARY) -lcudart -lcublas -lcurand -lstdc++   -lnvidia-ml -lNVMLWrapper

OBJ=matrixMul.o matrixMul_kernel.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard *.h) Makefile

all: obj  $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)  
	$(NVCC) $(COMMON) -Xcompiler "$(CXXFLAGS)" $^ -o $@ $(LDFLAGS) 

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON)  $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(COMMON) $(ARCH) -Xcompiler "$(CXXFLAGS)"  $(NVCCFLAGS) -c $< -o $@

obj:
	mkdir -p obj


clean:
	rm -rf $(EXEC) $(OBJDIR)/*
	
	
DEF_SIZE=128
BATCH_SIZE=32
DEF_A_INPUT=./mxm_A_$(DEF_SIZE)_$(BATCH_SIZE).matrix
DEF_B_INPUT=./mxm_B_$(DEF_SIZE)_$(BATCH_SIZE).matrix
DEF_GOLD=./mxm_GOLD_$(DEF_SIZE)_$(BATCH_SIZE).matrix

generate: $(EXEC)
	./$(EXEC) -generate -size=$(DEF_SIZE) -batch=$(BATCH_SIZE) -kernel_type=0 \
			-input_a=$(DEF_A_INPUT) \
			-input_b=$(DEF_B_INPUT) \
			 -gold=$(DEF_GOLD)

test: $(EXEC) 
	./$(EXEC) -size=$(DEF_SIZE) -batch=$(BATCH_SIZE) -kernel_type=0 \
			-input_a=$(DEF_A_INPUT) \
			-input_b=$(DEF_B_INPUT) \
			-gold=$(DEF_GOLD) \
			-iterations=10 -verbose

	
