INCLUDE=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I../common/include
all: cudaTensorCoreGEMM.cu
	nvcc -o TCGemm -arch=sm_70 -lcurand $(INCLUDE) cudaTensorCoreGEMM.cu

clean:
	rm -f TCGemm
