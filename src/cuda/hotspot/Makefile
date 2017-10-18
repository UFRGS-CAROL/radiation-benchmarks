CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH) #-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: hotspot

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) -DLOGS hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper $(NVCCFLAGS) -Xcompiler " -fopenmp" -o hotspot

clean:
	rm -f hotspot gold_*
