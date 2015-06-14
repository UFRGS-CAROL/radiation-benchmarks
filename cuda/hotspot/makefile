CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

MATRIX_NAMES=input_power_1024 input_temp_1024 output_1024

ALL: gen hotspot

gen: hotspot_generate.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_generate.cu -o hotspot_generate

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper -o hotspot

clean:
	rm hotspot_generate hotspot hotspot_ecc hotspot_thread $(MATRIX_NAMES)

