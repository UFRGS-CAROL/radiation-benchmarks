NVCC=nvcc
TARGET=lud
LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/
NVIDIA_LIB=../../include/
NVIDIA_INC=../../include/

ifeq ($(dbg), 1)
	DEBUG_FLAGS= -g -G
endif

LOG_FLAGS=
ifeq ($(logs), 1)
	LOG_FLAGS += -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelperCuda -DLOGS
endif 

GENCODE= -gencode arch=compute_35,code=sm_35  
NVCCFLAGS=-lcublas -lcublas_device -lcudadevrt $(GENCODE) -Xcompiler -fopenmp -I$(NVIDIA_LIB) -L$(NVIDIA_INC)


all: clean $(TARGET) generate

$(TARGET):lud.cu lud_kernel.cu
	$(NVCC) lud.cu lud_kernel.cu $(NVCCFLAGS) $(LOG_FLAGS) $(DEBUG_FLAGS) -o lud


test:

generate:


clean:
	rm -rf lud lud_gold_* lud_input_*
