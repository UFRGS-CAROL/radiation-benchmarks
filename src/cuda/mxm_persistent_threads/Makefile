LOGS=1
DEBUG?=0
FI=0
BUILDPROFILER=0

CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

#ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
ARCH+= -gencode arch=compute_35,code=[sm_35,compute_35]	# Titan V | Tesla V100
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72]	# Jetson Xavier

NVCCFLAGS= $(ARCH) -Xptxas -v -std=c++11  -O3 -D_FORCE_INLINES -dc $(EXTRA_NVCC_FLAGS)  $(EXTRA_LINK_FLAGS)
CXXFLAGS= -fopenmp -O3
INCLUDE= -I$(CUDAPATH)/include/ -I../../include/ -I../common/include/ -I$(CUDAPATH)/samples/common/inc/
LIBRARY= -L$(CUDAPATH)/lib64/  -L../../include/  -L/home/carol/radiation-benchmarks/src/cuda/common/lib -L$(CUDAPATH)/lib64/stubs

OBJDIR=./obj/

EXEC=cuda_batched_mxm

LD_LIB=LD_LIBRARY_PATH=/usr/local/sassi7/lib64/

ifeq ($(FI), 1)
FIDEBUG= -debug
endif

ifeq ($(DEBUG), 1)
NVCCFLAGS+= -g -G
CXXFLAGS+= -g
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LIBRARY+= -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
NVCCFLAGS+= -DBUILDPROFILER
LDFLAGS+= -lnvidia-ml -lNVMLWrapper
endif

COMMON= $(INCLUDE)
LDFLAGS= $(LIBRARY) -lcudart -lcublas -lcurand -lstdc++ 

OBJ=matrixMul.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard *.h) Makefile


all: clean obj  $(EXEC) #install

$(EXEC): $(OBJS) $(EXECOBJ)  
	$(NVCC) $(COMMON) -Xcompiler "$(CXXFLAGS)" $^ -o $@ $(LDFLAGS) $(ARCH) $(EXTRA_NVCC_FLAGS)  $(EXTRA_LINK_FLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON)  $(CXXFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(COMMON) $(ARCH) -Xcompiler "$(CXXFLAGS)"  $(NVCCFLAGS) -c $< -o $@

obj:
	mkdir -p obj


#install:
#	mkdir -p $(REGRESSION_HOME)/bin/$(OPTION)/$(SUITE_NAME)/
#	cp $(EXEC) $(REGRESSION_HOME)/bin/$(OPTION)/$(SUITE_NAME)/

clean:
	rm -rf $(EXEC) $(OBJDIR)/* 

DEF_SIZE=112
BATCH_SIZE=32
DEF_A_INPUT=`pwd`/mxm_A_$(DEF_SIZE)_$(BATCH_SIZE).matrix
DEF_B_INPUT=`pwd`/mxm_B_$(DEF_SIZE)_$(BATCH_SIZE).matrix
DEF_GOLD=`pwd`/mxm_GOLD_$(DEF_SIZE)_$(BATCH_SIZE).matrix
KERNELTYPE=1


golden: $(EXEC)
	./$(EXEC) -size=$(DEF_SIZE) -batch=$(BATCH_SIZE) -kernel_type=$(KERNELTYPE) \
			-input_a=$(DEF_A_INPUT) \
                        -input_b=$(DEF_B_INPUT) \
                        -gold=$(DEF_GOLD) \
			-iterations=1 > golden_stdout.txt 2>golden_stderr.txt



generate: $(EXEC)
	./$(EXEC) -generate -size=$(DEF_SIZE) -batch=$(BATCH_SIZE) -kernel_type=$(KERNELTYPE) \
			-input_a=$(DEF_A_INPUT) \
			-input_b=$(DEF_B_INPUT) \
			 -gold=$(DEF_GOLD) -verbose

test: $(EXEC) 
	./$(EXEC) -size=$(DEF_SIZE) -batch=$(BATCH_SIZE) -kernel_type=$(KERNELTYPE) \
			-input_a=$(DEF_A_INPUT) \
			-input_b=$(DEF_B_INPUT) \
			-gold=$(DEF_GOLD) \
			-iterations=1 $(FIDEBUG) 
	
