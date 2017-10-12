# 
#  Copyright (c) 2016 University of Cordoba and University of Illinois
#  All rights reserved.
# 
#  Developed by:    IMPACT Research Group
#                   University of Cordoba and University of Illinois
#                   http://impact.crhc.illinois.edu/
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  with the Software without restriction, including without limitation the 
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#  sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#       > Redistributions of source code must retain the above copyright notice,
#         this list of conditions and the following disclaimers.
#       > Redistributions in binary form must reproduce the above copyright
#         notice, this list of conditions and the following disclaimers in the
#         documentation and/or other materials provided with the distribution.
#       > Neither the names of IMPACT Research Group, University of Cordoba, 
#         University of Illinois nor the names of its contributors may be used 
#         to endorse or promote products derived from this Software without 
#         specific prior written permission.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
#  CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
#  THE SOFTWARE.
# 
CUDA_DIR = /usr/local/cuda

CXX=g++
NVCC=$(CUDA_DIR)/bin/nvcc
NVCC_FLAGS=-std=c++11
CXX_FLAGS= -std=c++11 -lcudart -lcublas -lcurand -pthread 

CHAI_CUDA_LIB=/usr/local/cuda-8.0/lib64
CHAI_CUDA_INC=/usr/local/cuda-8.0/include

ARCH= -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 \
	  -gencode=arch=compute_60,code=sm_60 \
      -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62

NVCC_FLAGS+= $(ARCH)

ifeq ($(UNIFIEDMEM), 1)
NVCC_FLAGS+= -DCUDA_8_0
endif



ifndef CHAI_CUDA_LIB
    $(error CHAI_CUDA_LIB not defined. This environment variable must be defined to point to the location of the CUDA library)
endif
LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm

ifndef CHAI_CUDA_INC
    $(error CHAI_CUDA_INC not defined. This environment variable must be defined to point to the location of the CUDA header files)
endif
INC=-I$(CHAI_CUDA_INC) -I./support

ifeq ($(LOGS), 1)
INC+= -I../../include/
LIB+= -L../../include -lLogHelper -DLOGS
endif


DEP=kernel.cpp kernel.h main.cpp kernel.cu support/common.h support/cuda-setup.h support/partitioner.h support/timer.h support/verify.h
SRC=logs_processing.cpp main.cpp kernel.cpp kernel.cu 
EXE=bfs

OBJDIR=./obj/

RAD_BENCH=/home/carol/radiation-benchmarks


OBJ= kernel_cxx.o logs_processing.o kernel.o main.o

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: make_obj $(EXE)

$(EXE): $(OBJS)
	$(CXX)  $^ $(LIB) $(INC) -o $@  $(CXX_FLAGS)


$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXX_FLAGS) -c $< -o $@  $(LIB) $(INC)

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

test:
	$(RAD_BENCH)/src/cuda/bezier_surface/$(EXE) 

generate:
	$(RAD_BENCH)/src/cuda/bezier_surface/$(EXE) 


make_obj:
	mkdir -p $(OBJDIR)


clean:
	rm -f $(EXE) $(OBJDIR)*.o 

