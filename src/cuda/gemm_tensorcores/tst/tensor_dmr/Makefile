CXX=g++
CXXFLAGS= -std=c++11 -o3  -fPIC 
EXEC=mxm-tensor

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCCFLAGS= -std=c++11 -O3 -Xptxas -v


ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

INCLUDE=-I./include -I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I$(RAD_DIR)/src/cuda/common/include

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt -lpthread

DEPS = Makefile

all: $(EXEC)
	
$(EXEC): main.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) $< -o $@ $(INCLUDE) 

clean:
	rm -rf $(OBJS) $(TARGET)
	