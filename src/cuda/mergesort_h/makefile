CC=g++
NVCC=nvcc

NVIDIA_IC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: mergesort

mergesort: bitonic.cu main.cpp mergeSort.cu mergeSort_host.cpp mergeSort_validate.cpp verifySort.cu
	$(NVCC) $(NVCC_FLAGS) -g bitonic.cu main.cpp mergeSort.cu mergeSort_host.cpp mergeSort_validate.cpp verifySort.cu -arch=sm_35 -I$(LOGHELPER_INC),./inc/ -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda $(NVCCFLAGS) -Xcompiler "-O3 -fopenmp" -o mergesort

clean:
	rm -rf mergesort mergesort_*
