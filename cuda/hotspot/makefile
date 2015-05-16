CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch=sm_35

NVIDIA_INC=/usr/local/cuda/include/
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

MATRIX_NAMES=input_power1024 input_temp1024 output_1024

ALL: gen hotspot ecc hard_thread

gen: hotspot_generate.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_generate.cu -o hotspot_generate

hotspot: hotspot.cu
	$(NVCC) $(NVCC_FLAGS) hotspot.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -o hotspot

ecc: hotspot_ecc.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_ecc.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -o hotspot_ecc

hard_thread: hotspot_hard_thread.cu
	$(NVCC) $(NVCC_FLAGS) hotspot_hard_thread.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -o hotspot_thread

clean:
	rm hotspot_generate hotspot hotspot_ecc hotspot_thread $(MATRIX_NAMES)

