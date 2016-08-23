CC=g++
NVCC=nvcc

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: radixsort

radixsort: radixSortThrust.cu sortVerify.cu
	$(NVCC) $(NVCC_FLAGS) sortVerify.cu radixSortThrust.cu -arch=sm_35 -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda -lHelperSorts -Xcompiler "-fopenmp -O3" -o radixsort

clean:
	rm -rf radixsort radixsort_*
