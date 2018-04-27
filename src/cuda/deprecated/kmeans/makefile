# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

CUDA_LIB_DIR=/usr/local/cuda/lib64
LOGHELPER_INC=../../include
LOGHELPER_LIB=../../include

# CUDA compiler
NVCC = nvcc
NVCC_FLAGS = -arch=sm_35

all : kmeans kmeans_generate

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans -L$(CUDA_LIB_DIR) -L$(LOGHELPER_LIB) $(NVCCFLAGS) -lLogHelperCuda -lcudart -lm

kmeans_generate: cluster.o getopt.o kmeans.o kmeans_clustering.o kmeans_cuda.o rmse.o kmeans_generate.o
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans_generate.o kmeans_clustering.o kmeans_cuda.o rmse.o -o kmeans_generate -L$(CUDA_LIB_DIR) -lcudart -lm

kmeans.o: kmeans.c
	$(CC) $(CC_FLAGS) kmeans.c -c -I$(LOGHELPER_INC)

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

kmeans_cuda.o: kmeans_cuda.cu 
	$(NVCC) $(NVCC_FLAGS) -c kmeans_cuda.cu

clean:
		rm -f *.o *~ kmeans kmeans_generate kmeans_cuda.linkinfo input_* output_*
