ifeq ($(SINGLEITERACTION), 1)  #at this point, the makefile checks if FEATURE is enabled
ITER = -DITERACTIONS=1 #variable passed to g++
endif

CC=g++
NVCC=nvcc
NVCC_FLAGS=-arch sm_35

LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=lavaMD_192_13.cu
SOURCES_GEN=lavaMD_generate.cu
RUN_EXEC=lava
GEN_EXEC=lava_generate

MATRIX_NAMES=output_forces_192_13 input_distances_192_13 input_charges_192_13

ALL: gen lava ecc

gen: lavaMD_generate.cu
	$(NVCC) $(NVCC_FLAGS) lavaMD_generate.cu -o lava_generate

lava: lavaMD_192_13.cu
	$(NVCC) $(NVCC_FLAGS) lavaMD_192_13.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) $(ITER) -lLogHelper -o lava

ecc: lavaMD_192_13_ecc.cu
	$(NVCC) $(NVCC_FLAGS) lavaMD_192_13_ecc.cu -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -o lava_ecc

clean:
	rm lava_generate lava lava_ecc $(MATRIX_NAMES)

