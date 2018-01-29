CC=g++
NVCC=/usr/local/cuda/bin/nvcc
ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH) #-arch=sm_35

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/
NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/

RUN_EXEC=hlava
GEN_EXEC=hlava_generate

MATRIX_NAMES=output_gold_* input_distances_* input_charges_*

ALL: hlava

hlava: hlava.cu
	$(NVCC) $(NVCC_FLAGS) -DLOGS $(DOUBLE_PRECISION) hlava.cu -I$(LOGHELPER_INC) -I$(NVIDIA_INC) -L$(LOGHELPER_LIB) -L$(NVIDIA_LIB) -lLogHelper -Xcompiler " -fopenmp" -o hlava

clean:
	rm -f hlava_generate hlava $(MATRIX_NAMES)
