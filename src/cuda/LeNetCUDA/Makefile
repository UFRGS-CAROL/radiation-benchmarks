EXEC=leNetCUDA

SRC=./src/
NVCC=nvcc
CXX=g++

OBJDIR=./obj/

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62] 
      
      

#for radiation setup
LOGS?=0
COMMON=

CFLAGS= -Ofast -std=c++11


ifeq ($(DEBUG),1)
OPTS= -G -g
endif

ifeq ($(LOGS), 1)
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/
endif

GPU_KERNELS=ConvKernels.o


all: obj $(EXEC)

$(EXEC): $(OBJDIR)$(GPU_KERNELS)
	$(CXX) $(COMMON) $(CFLAGS) $(OBJDIR)$(GPU_KERNELS) $(SRC)main.cpp $(LOGHELPER_INC) $(LOGHELPER_LIB) -o $(EXEC)


$(OBJDIR)$(GPU_KERNELS): 
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $(SRC)ConvolutionalLayer.cu -o $(OBJDIR)$(GPU_KERNELS)
	


obj:
	mkdir -p obj
	
.PHONY: clean
clean:
	rm -rf $(OBJS) $(EXEC)