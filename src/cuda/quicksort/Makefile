CXX=g++
CXXFLAGS= -std=c++11 -O3  -Xcompiler -fPIC  -Xcompiler  -pthread
EXEC=quicksort
LOGS=1
BUILDPROFILER=0
DEBUG=0
SIZE=134217728

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3 -Xptxas -v -rdc=true


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

OBJDIR=./obj/
OBJ= cdpBitonicSort.o cdpAdvancedQuicksort.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
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
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS) $(INCLUDE) $(CXXFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
	
generate:
	 ./$(EXEC) -size=$(SIZE) -generate -verbose -debug \
	 		   -input=./input_$(SIZE) \
	 		   -gold=./gold_$(SIZE) \
	 		   -iterations=1 -noinputensurance
	 		   	
test:
	 ./$(EXEC) -size=$(SIZE) -verbose -debug \
	 		   -input=./input_$(SIZE) \
	 		   -gold=./gold_$(SIZE) \
	 		   -iterations=1 -noinputensurance
