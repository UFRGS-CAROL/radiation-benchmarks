CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: hotspot

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper $(NVCCFLAGS) -Xcompiler " -fopenmp" -o hotspot

clean:
	rm -f hotspot gold_*
