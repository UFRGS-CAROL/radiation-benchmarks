CC=g++
NVCC=/usr/local/cuda/bin/nvcc
PRECISION=single
LOWERCASED_PRECISION=single

HARDENING_NAME=$(if $(HARDENING),hard,unhard)

ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100

NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

SOURCES=./cuda_mxm-hard.cu
EXEC_RADIX=./cuda_mxm-$(HARDENING_NAME)_
EXEC=$(EXEC_RADIX)$(LOWERCASED_PRECISION)

DEF_SIZE=1024

DEF_A_INPUT=./mxm_$(LOWERCASED_PRECISION)_A_8192.matrix
DEF_B_INPUT=./mxm_$(LOWERCASED_PRECISION)_B_8192.matrix
DEF_GOLD=./mxm_$(LOWERCASED_PRECISION)_GOLD_$(DEF_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./mxm*.matrix

all: $(EXEC)

check-benchtype:
	echo "syntax: make [HARDENING=whatevernotnil]"

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -Dtest_precision_$(LOWERCASED_PRECISION) $(if $(HARDENING),-DHARDENING,) -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS) -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)* $(MATRIX_NAMES)

generate: $(EXEC)
	$(EXEC) -generate -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: $(EXEC) generate
	$(EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=10 -verbose
