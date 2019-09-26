CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -pthread 
EXEC=cudaCacheTest
DISABLEL1=0
LOGS=1
BUILDPROFILER=0

VPATH=./src/
#CUDAPATH=/home/carol/cuda/cuda-toolkit
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3 -Xptxas -v


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
#ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

OBJDIR=./obj/
OBJ= l1_cache_kernels.o register_file_kernels.o main.o 
OBJ+=  shared_memory_kernels.o  read_only_cache_kernels.o l2_cache_kernels.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(DISABLEL1), 1)
NVCCFLAGS+= -Xptxas -dlcm=cg
CXXFLAGS+= -DL2TEST
EXEC=cudaCacheTestL2
else
NVCCFLAGS+= -Xptxas -dlcm=ca
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L../../include  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L../common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC) cudaCacheTestL2
	
	
test:
	./$(EXEC) --iterations 10
	
testl2:
	./$(EXEC)L2 --iterations 10 --memtotest L2

testshared:
	./$(EXEC) --iterations 10 --memtotest SHARED
	
testregisters:
	env LD_LIBRARY_PATH=/home/carol/cuda/cuda-toolkit/lib64 ./$(EXEC) --iterations 10 --memtotest REGISTERS	
	
testconstant:
	env LD_LIBRARY_PATH=/home/carol/cuda/cuda-toolkit/lib64 ./$(EXEC) --iterations 10 --memtotest READONLY	
	
