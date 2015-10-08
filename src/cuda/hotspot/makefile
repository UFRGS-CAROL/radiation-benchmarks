CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

MATRIX_NAMES=output_*

all: hotspot hotspot_generate

hotspot_generate: hotspot_generate.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_generate.cu -o hotspot_generate

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda $(NVCCFLAGS) -Xcompiler " -fopenmp" -o hotspot

clean:
	rm -f hotspot_generate hotspot $(MATRIX_NAMES)
