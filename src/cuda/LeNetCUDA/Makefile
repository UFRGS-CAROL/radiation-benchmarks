VPATH=./src/
EXEC=leNetCUDA
OBJDIR=./obj/

SRC=./src/
NVCC=nvcc
CXX=g++

OBJDIR=./obj/

GPU?=0

CUDA_THRUST=
#-I/usr/local/cuda/include/thrust/

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62] 
      
#for radiation setup
LOGS?=0
COMMON=-std=c++11 


CFLAGS= $(COMMON) 

NVCC_FLAGS= -Xcompiler -Ofast
#-std=c++11 

ifeq ($(DEBUG),1)
OPTS= -g
endif

ifeq ($(LOGS), 1)
LOGHELPER_INC=-I../../include/
LOGHELPER_LIB=-I../../include/
endif

ifeq ($(GPU), 1)
ON_GPU= -DGPU
CFLAGS+=  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
ifeq ($(DEBUG),1)
NVCC_FLAGS+= -G
endif

endif

GPU_KERNELS=ConvKernels.o

OBJ=Layer.o ConvolutionalLayer.o MaxpoolingLayer.o FullyConnectedLayer.o \
OutputLayer.o ConvNet.o MNISTParser.o Util.o main.o

ifeq ($(GPU), 1)
OBJ+= ConvolutionalLayerKernel.o MaxpoolingLayerKernel.o \
FullyConnectedLayerKernel.o  OutputLayerKernel.o cudaUtil.o
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

all: obj $(EXEC)

$(EXEC): $(OBJS)
	$(CXX)  $^ $(LOGHELPER_INC) $(LOGHELPER_LIB) $(ON_GPU) -o $@  $(LDFLAGS)  $(CFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CFLAGS) $(OPTS) $(ON_GPU) -c $< -o $@  $(HELPFUL) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(OPTS) $(NVCC_FLAGS) -c $< -o $@



debug: 
	$(NVCC) $(ARCH) $(COMMON) $(OPTS) src/debug.cu -o debug 

train:
	./$(EXEC) train train-images-idx3-ubyte train-labels-idx1-ubyte ./lenet.weights

test:
	./$(EXEC) classify t10k-images-idx3-ubyte t10k-labels-idx1-ubyte ./lenet.weights

obj:
	mkdir -p obj
	
.PHONY: clean
clean:
	rm -rf obj/*o $(EXEC) debug
