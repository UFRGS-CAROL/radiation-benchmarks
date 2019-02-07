CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -pthread 
EXEC=cudaCacheTest
DISABLEL1CACHE=0
LOGS=0

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include

OBJDIR=./obj/
OBJ= NVMLWrapper.o utils.o l1_cache_kernels.o l2_cache_kernels.o register_file_kernels.o
OBJ+= read_only_cache_kernels.o shared_memory_kernels.o main.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -G
endif

ifeq ($(DISABLEL1CACHE), 1)
NVCCFLAGS+= -Xptxas -dlcm=cg
CXXFLAGS+= -DL2TEST
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
LDFLAGS+= -L../../include  -lLogHelper
endif


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -L$(CUDAPATH)/lib64/stubs/ -lnvidia-ml 

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
	rm -f $(OBJDIR)* $(EXEC)
	
	
test:
	./$(EXEC) --iterations 10
	