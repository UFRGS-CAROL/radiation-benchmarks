CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch sm_35

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_EXEC=lava
GEN_EXEC=lava_generate

MATRIX_NAMES=output_* input_distances_* input_charges_*

ALL: lava lava_generate

lava_generate: lava_generate.cu
	$(NVCC) $(NVCC_FLAGS) lava_generate.cu -o lava_generate

lava: lava.cu
	$(NVCC) $(NVCC_FLAGS) lava.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -Xcompiler " -fopenmp" -o lava

clean:
	rm -f lava_generate lava $(MATRIX_NAMES)
