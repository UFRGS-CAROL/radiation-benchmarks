CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=compute_53,sm_53

NVIDIA_INC=/usr/local/cuda/include/;/usr/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: hotspot

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) -DLOGS hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelperCuda $(NVCCFLAGS) -Xcompiler " -fopenmp" -o hotspot

clean:
	rm -f hotspot gold_*
