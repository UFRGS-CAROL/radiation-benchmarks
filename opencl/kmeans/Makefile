#include ../../common/make.config
# C compiler
CC = g++
CC_FLAGS = -g -O2 

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

OPENCL_INC=/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
OPENCL_LIB=/opt/intel/opencl-1.2-4.5.0.8/lib64/


SOURCES=$(wildcard *.c)


amd: amd_check

amd_check: $(SOURCES)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL $(SOURCES) -o kmeans $(AMD_INC) $(AMD_LIB)



#kmeans: cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o
#	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL kmeans.cpp cluster.o getopt.o read_input.o kmeans_clustering.o rmse.o -o kmeans -I$(OPENCL_INC) -L$(OPENCL_LIB)
#
#%.o: %.[ch]
#	$(CC) $(CC_FLAGS) $< -c
#
#cluster.o: cluster.c 
#	$(CC) $(CC_FLAGS) cluster.c -c
#
#getopt.o: getopt.c 
#	$(CC) $(CC_FLAGS) getopt.c -c
#
#kmeans.o: kmeans.c 
#	$(CC) $(CC_FLAGS) read_input.c -c
#
#rmse.o: rmse.c
#	$(CC) $(CC_FLAGS) rmse.c -c

clean:
	rm -f *.o *~ kmeans *.linkinfo
