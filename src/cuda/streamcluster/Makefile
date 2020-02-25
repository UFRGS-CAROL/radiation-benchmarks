CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC -fopenmp 
EXEC=cudaSC
LOGS=1
BUILDPROFILER=0
ITERATIONS=10
GOLD=./gold.data
INPUT=./input.data
DIMSIZE=1024

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc


NVCCFLAGS= -std=c++11 -O3 -Xptxas -v  -Wno-deprecated-gpu-targets


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I../../include -I../common/include

OBJDIR=./obj/
OBJ= streamcluster_cpu.o streamcluster_cuda.o Parameters.o


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


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

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
	

generate:
	./$(EXEC) --k1 10 --k2 20 --d $(DIMSIZE) --n 65536 \
			  --chunksize 65536 --clustersize 1000 \
			  --input $(INPUT) --gold $(GOLD) --generate --verbose

	
test:
	./$(EXEC) --k1 10 --k2 20 --d $(DIMSIZE) --n 65536 \
			  --chunksize 65536 --clustersize 1000 \
			  --input $(INPUT) --gold $(GOLD) \
			  --generate --verbose --iterations 10

	
