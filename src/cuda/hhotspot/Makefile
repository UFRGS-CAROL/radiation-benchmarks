CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_53

NVIDIA_INC=/usr/local/cuda/include/;/usr/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: hhotspot

hhotspot: hhotspot.cu
	$(NVCC) $(NVCC_FLAGS) -DLOGS hhotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper -Xcompiler " -fopenmp" -o hhotspot

clean:
	rm -f hhotspot gold_*
