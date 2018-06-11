CC=g++
NVCC=/usr/local/cuda/bin/nvcc

LOWERCASED_TYPE=$(shell echo $(TYPE) | tr A-Z a-z)

ifeq ($(LOWERCASED_TYPE),half)
ARCH= -gencode arch=compute_62,code=[sm_62,compute_62] \
      -gencode arch=compute_70,code=[sm_70,compute_70]
else
ARCH= -gencode arch=compute_30,code=[sm_30,compute_30] \
      -gencode arch=compute_35,code=[sm_35,compute_35] \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62] \
      -gencode arch=compute_70,code=[sm_70,compute_70]
endif


NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=./cuda_trip_micro-fma.cu
EXEC_RADIX=./cuda_trip_micro-fma_
EXEC=$(EXEC_RADIX)$(LOWERCASED_TYPE)

all: $(EXEC)

check-benchtype:
	echo "syntax: make TYPE=<double|half|single>"
	test $(TYPE)

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -Dtest_type_$(LOWERCASED_TYPE) -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS) -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)*

test: $(EXEC)
	$(EXEC) -verbose -iterations=10
