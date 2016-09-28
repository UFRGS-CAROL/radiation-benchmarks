CC=g++
NVCC=nvcc

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: radixsort_h

radixsort_h: radixSortThrust.cu sortVerify.cu
	$(NVCC) $(NVCC_FLAGS) sortVerify.cu radixSortThrust.cu -arch=sm_35 -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda -Xcompiler "-fopenmp -O3" -o radixsort_h

clean:
	rm -rf radixsort_h radixsort_*
