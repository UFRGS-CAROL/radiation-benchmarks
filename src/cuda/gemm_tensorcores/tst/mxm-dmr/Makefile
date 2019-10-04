CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC 
EXEC=mxm-dmr

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -O3 -Xptxas -v

#ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
#ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62] #Tegra X2
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V
#ARCH+= -gencode arch=compute_72,code=[sm_72,compute_72] #XavierV
INCLUDE= -I./include -I$(CUDAPATH)/include -I/home/fernando/radiation-benchmarks/src/cuda/common/include/

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt -lpthread

DEPS = $(wildcard *.h) Makefile

all: $(EXEC)
	
$(EXEC): main.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) $< -o $@ $(INCLUDE) 

clean:
	rm -f $(OBJDIR)* $(EXEC)
