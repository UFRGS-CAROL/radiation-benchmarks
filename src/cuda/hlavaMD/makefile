CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_53

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
