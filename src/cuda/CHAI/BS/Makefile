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

ARCH=  -gencode arch=compute_30,code=[sm_30,compute_30] 
ARCH+= -gencode arch=compute_35,code=[sm_35,compute_35]
ARCH+= -gencode arch=compute_50,code=[sm_50,compute_50] 
ARCH+= -gencode arch=compute_52,code=[sm_52,compute_52] 
ARCH+= -gencode arch=compute_60,code=[sm_60,compute_60] 
ARCH+= -gencode arch=compute_62,code=[sm_62,compute_62]

NVCC=nvcc
CXX=g++
CXX_FLAGS=-std=c++11 $(ARCH)
LOGS?=1

CHAI_CUDA_LIB?=/usr/local/cuda/lib64
CHAI_CUDA_INC?=/usr/local/cuda/include


LOGHELPERPATH=../../../include/


#ifndef CHAI_CUDA_LIB
#    $(error CHAI_CUDA_LIB not defined. This environment variable must be defined to point to the location of the CUDA library)
#endif
#LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -L/home/carol/radiation-benchmarks/src/include/ -lLogHelper -DLOGS -lm
LIB=-L/usr/lib/ -L$(CHAI_CUDA_LIB) -lm


#ifndef CHAI_CUDA_INC
#    $(error CHAI_CUDA_INC not defined. This environment variable must be defined to point to the location of the CUDA header files)
#endif
#INC=-I$(CHAI_CUDA_INC) -I/home/carol/radiation-benchmarks/src/cuda/CHAI/BS/ -I/home/carol/radiation-benchmarks/src/include/ -I./support 
INC=-I$(CHAI_CUDA_INC) -I../BS/  -I./support 

ifeq ($(LOGS), 1)
LIB+= -L$(LOGHELPERPATH) -lLogHelper -DLOGS
INC+= -I$(LOGHELPERPATH)
endif 


DEP=kernel.cpp kernel.h main.cpp kernel.cu support/common.h support/cuda-setup.h support/partitioner.h support/timer.h support/verify.h
SRC=main.cpp kernel.cpp kernel.cu 
EXE=bs

all: $(EXE) gen_gold gen_input

$(EXE):
	$(NVCC) $(CXX_FLAGS) $(SRC) $(LIB) $(INC) -o $(EXE)
	
gen_gold:
	$(NVCC) $(CXX_FLAGS) bs_gen_gold.cpp -o gen_gold
	
	
gen_input:
	$(NVCC) $(CXX_FLAGS) bs_gen_input.cpp -o gen_input

clean:
	rm -f $(EXE) gen_input gen_gold

