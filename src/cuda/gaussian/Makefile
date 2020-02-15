#include ../../common/make.config

CC := nvcc

INCLUDE := /usr/local/cuda/include

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L/usr/local/cuda/lib64

clean:
	rm gaussian
