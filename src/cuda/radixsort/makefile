CC=g++
NVCC=nvcc

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: radixsort

radixsort: radixSortThrust.cu
	$(NVCC) $(NVCC_FLAGS) radixSortThrust.cu -arch=sm_35 -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda $(NVCCFLAGS) -Xcompiler "-O3 -fopenmp" -o radixsort

clean:
	rm -rf radixsort
