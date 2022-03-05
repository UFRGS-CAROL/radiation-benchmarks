CXX=g++
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

NVCC_FLAGS= -ccbin $(CXX)

NVIDIA_INC=$(CUDAPATH)/include/
NVIDIA_LIB=$(CUDAPATH)/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: radixsort

radixsort: radixSortThrust.cu
	$(NVCC) $(NVCC_FLAGS) $(logs) radixSortThrust.cu -arch=sm_35 -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper -Xcompiler "-fopenmp -O3" -o radixsort

clean:
	rm -rf radixsort radixsort_*
