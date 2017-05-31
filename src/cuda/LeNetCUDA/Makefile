EXEC=leNetCUDA

SRC=./src/
NVCC=nvcc
CXX=g++

OBJDIR=./obj/



CUDA_THRUST=
#-I/usr/local/cuda/include/thrust/

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62] 
      
OBJS=$(OBJDIR)/*.o

#for radiation setup
LOGS?=0
COMMON=


CFLAGS= -std=c++11  -L/usr/local/cuda/lib64 -lcuda -lcudart

NVCC_FLAGS=-std=c++11 

ifeq ($(DEBUG),1)
OPTS= -G -g
endif

ifeq ($(LOGS), 1)
LOGHELPER_INC=-I../../include/
LOGHELPER_LIB=-I../../include/
endif

GPU_KERNELS=ConvKernels.o


all: obj $(EXEC)

$(EXEC): $(OBJDIR)$(GPU_KERNELS)
	$(NVCC) $(COMMON) $(NVCC_FLAGS) $(SRC)main.cpp $(LOGHELPER_INC) $(LOGHELPER_LIB) -o $(OBJDIR)main.o -c
	$(NVCC) $(COMMON) $(NVCC_FLAGS) $(OBJDIR)$(GPU_KERNELS) $(OBJDIR)main.o $(LOGHELPER_INC) $(LOGHELPER_LIB) $(CUDA_THRUST) -o $(EXEC)


$(OBJDIR)$(GPU_KERNELS): 
	$(NVCC) $(ARCH) $(NVCC_FLAGS) $(COMMON) -c $(SRC)ConvolutionalLayer.cu -o $(OBJDIR)$(GPU_KERNELS)
	


obj:
	mkdir -p obj
	
.PHONY: clean
clean:
	rm -rf $(OBJS) $(EXEC)
