GPU?=1
DEBUG?=0
#for radiation setup
LOGS?=0
DEBUG_ON_TITAN?=1

#
#ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52

VPATH=./src/
TARGET=gemm_wmma
OBJDIR=./obj/

CC=gcc
CXX=g++
NVCC=/usr/local/cuda/bin/nvcc 
OPTS=-Ofast
LDFLAGS= -lm -pthread -lstdc++ 
COMMON= 
CFLAGS=-Wall -Wfatal-errors  -Wunknown-pragmas


INCLUDE=-I/usr/local/cuda/include

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
NVCCFLAGS+=-G -DDEBUG
endif

CFLAGS+=$(OPTS)
STDVERSION=--std=c++11

COMMON+= $(STDVERSION)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
endif

OBJ= main.o 

ifeq ($(SAFE_MALLOC), 1)
CFLAGS+= -DSAFE_MALLOC
endif

ifeq ($(LOGS), 1)
INCLUDE+=-I../../include/
NVCCFLAGS+= -DLOGS=1
LDFLAGS+= -L../../include/ -lLogHelper -DLOGS=1
endif

ifeq ($(DEBUG_ON_TITAN), 1)
CFLAGS+= -DDEBUG_ON_TITAN
endif
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./*.h) Makefile

RAD_DIR=/home/carol/radiation-benchmarks


all: obj $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

#$(OBJDIR)%.o: %.c $(DEPS)
#	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@  $(HELPFUL) $(LOGHELPER_LIB) $(LOGHELPER_INC)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(INCLUDE) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $(INCLUDE) -c $< -o $@ 


obj:
	mkdir -p obj


.PHONY: clean
clean:
	rm -rf $(OBJS) $(TARGET) ./*.matrix




generate:
	./$(TARGET) --size 1024 --generate 1 --verbose 1 --tensor_cores 1
	
test:
	./$(TARGET) --size 1024 --generate 0 --verbose 1 --iterations 10 --tensor_cores 0
	
