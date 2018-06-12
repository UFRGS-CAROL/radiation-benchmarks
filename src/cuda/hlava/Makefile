CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_53,code=[sm_53,compute_53] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \	# Tesla P100
      -gencode arch=compute_61,code=[sm_61,compute_61] \	# Titan X | Titan Xp
      -gencode arch=compute_62,code=[sm_62,compute_62] \	# Tegra X2
      -gencode arch=compute_70,code=[sm_70,compute_70]		# Titan V | Tesla V100


NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_SOURCES=./hlava.cu
RUN_EXEC=./hlava

DEF_SIZE=12

DEF_CHARGES_INPUT=./hlava_charges_$(DEF_SIZE)
DEF_DISTANCES_INPUT=./hlava_distances_$(DEF_SIZE)
DEF_GOLD=./hlava_gold_$(DEF_SIZE)

INPUTS_NAMES=$(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD) ./hlava_charges_* ./hlava_distances_* ./hlava_gold_*

ALL: $(RUN_EXEC)

$(RUN_EXEC): $(RUN_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -DLOGS $(RUN_SOURCES) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -Xcompiler " -fopenmp" -o $(RUN_EXEC)

clean:
	rm -f $(RUN_EXEC) $(INPUTS_NAMES)

generate: $(RUN_EXEC)
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -generate -streams=1 -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -streams=1 -iterations=10 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
