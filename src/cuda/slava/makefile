CC=g++
NVCC=/usr/local/cuda/bin/nvcc

ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_35,code=sm_35 \
      -gencode arch=compute_50,code=[sm_50,compute_50] \
      -gencode arch=compute_52,code=[sm_52,compute_52] \
      -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_62,code=[sm_62,compute_62]


NVCC_FLAGS= $(ARCH)

NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

RUN_SOURCES=./slava.cu
RUN_EXEC=./slava

DEF_SIZE=12

DEF_CHARGES_INPUT=./slava_charges_$(DEF_SIZE)
DEF_DISTANCES_INPUT=./slava_distances_$(DEF_SIZE)
DEF_GOLD=./slava_gold_$(DEF_SIZE)

INPUTS_NAMES=$(DEF_CHARGES_INPUT) $(DEF_DISTANCES_INPUT) $(DEF_GOLD) ./slava_charges_* ./slava_distances_* ./slava_gold_*

ALL: $(RUN_EXEC)

$(RUN_EXEC): $(RUN_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -DLOGS $(RUN_SOURCES) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper -Xcompiler " -fopenmp" -o $(RUN_EXEC)

clean:
	rm -f $(RUN_EXEC) $(INPUTS_NAMES)

generate: $(RUN_EXEC)
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -generate -streams=1 -iterations=1 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)

test: $(RUN_EXEC) generate
	$(RUN_EXEC) -boxes=$(DEF_SIZE) -streams=1 -iterations=10 -verbose -input_distances=$(DEF_DISTANCES_INPUT) -input_charges=$(DEF_CHARGES_INPUT) -output_gold=$(DEF_GOLD)
