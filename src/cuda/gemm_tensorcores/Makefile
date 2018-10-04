GPU?=1
DEBUG?=0
#for radiation setup
LOGS?=0

#ARCH=  -gencode arch=compute_30,code=[sm_30,compute_30] 
#ARCH+= -gencode arch=compute_35,code=[sm_35,compute_35]
#ARCH= -gencode arch=compute_50,code=[sm_50,compute_50] 
#ARCH+= -gencode arch=compute_52,code=[sm_52,compute_52] 
#ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] 
#ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62]
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]

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
CFLAGS=-Wall -Wfatal-errors 


INCLUDE=-I/usr/local/cuda/include

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g -DDEBUG
NVCCFLAGS=-G -DDEBUG
endif

CFLAGS+=$(OPTS)
STDVERSION=--std=c++11

COMMON+= $(STDVERSION)

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand
endif

OBJ=main.o GEMMWMMA.o

ifeq ($(SAFE_MALLOC), 1)
CFLAGS+= -DSAFE_MALLOC
endif

ifeq ($(LOGS), 1)
INCLUDE+=-I../../include/
LDFLAGS+= -L../../include/ -lLogHelper -DLOGS=1
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard ./*.cuh) Makefile

RAD_DIR=/home/carol/radiation-benchmarks


all: obj $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

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
	rm -rf $(OBJS) $(TARGET)




generate:
	#TODO:
	
test:
	#TODO:
