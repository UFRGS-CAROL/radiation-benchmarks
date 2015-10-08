LOGHELPER_INC=../../include/
LOGHELPER_LIB=../../include/

all: lud

lud: lud.cu lud_kernel.cu
	nvcc lud.cu lud_kernel.cu $(NVCCFLAGS) -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -Xcompiler -fopenmp -lcublas -lcublas_device -lcudadevrt -arch=sm_35 -o lud

clean:
	rm -rf lud lud_gold_* lud_input_*
