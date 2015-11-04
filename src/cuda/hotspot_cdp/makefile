CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: hotspot_cdp

hotspot_cdp: hotspot_cdp.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_cdp.cu -rdc=true -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda $(NVCCFLAGS) -Xcompiler " -fopenmp" -o hotspot_cdp

clean:
	rm -f hotspot_cdp
