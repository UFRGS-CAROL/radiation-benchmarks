VPATH=./src/
EXEC=leNetCUDA
OBJDIR=./obj/

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
      
#for radiation setup
LOGS?=0
COMMON=-std=c++11 


CFLAGS= $(COMMON)  -L/usr/local/cuda/lib64 -lcuda -lcudart

NVCC_FLAGS=
#-std=c++11 

ifeq ($(DEBUG),1)
OPTS= -G -g
endif

ifeq ($(LOGS), 1)
LOGHELPER_INC=-I../../include/
LOGHELPER_LIB=-I../../include/
endif

GPU_KERNELS=ConvKernels.o

OBJ=Layer.o ConvolutionalLayer.o ConvolutionalLayerKernel.o ConvNet.o \
FullyConnectedLayer.o MaxpoolingLayer.o MNISTParser.o OutputLayer.o Util.o main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CXX)  $^ $(LOGHELPER_INC) $(LOGHELPER_LIB) -o $@  $(LDFLAGS)  $(CFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(NVCC) $(ARCH) $(CFLAGS) -c $< -o $@  $(HELPFUL) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(NVCC_FLAGS)" -c $< -o $@ 

obj:
	mkdir -p obj
	
.PHONY: clean
clean:
	rm -rf $(OBJS) $(EXEC)
