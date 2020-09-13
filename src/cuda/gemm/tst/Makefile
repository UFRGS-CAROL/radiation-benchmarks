INCLUDE=-I/usr/local/cuda/include -I/usr/local/cuda/samples/common/inc -I../common/include
all: cudaTensorCoreGEMM.cu
	nvcc -o TCGemm -arch=sm_70 -lcurand $(INCLUDE) cudaTensorCoreGEMM.cu

mxm:
	nvcc -o mxm -arch=sm_70 $(INCLUDE) matrixMul.cu


clean:
	rm -f TCGemm mxm


test:
	./mxm -wA=4096 -wB=4096 -hA=4096 -hB=4096 -hA=4096
