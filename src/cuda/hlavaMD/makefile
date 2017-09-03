CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_53

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/
NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/

RUN_EXEC=lava
GEN_EXEC=lava_generate

MATRIX_NAMES=output_gold_* input_distances_* input_charges_*

ALL: lava

lava: lava.cu
	$(NVCC) $(NVCCFLAGS) -DLOGS lava.cu -I$(LOGHELPER_INC) -I$(NVIDIA_INC) -L$(LOGHELPER_LIB) -L$(NVIDIA_LIB) -lLogHelper -Xcompiler " -fopenmp" -o lava

clean:
	rm -f lava_generate lava $(MATRIX_NAMES)
