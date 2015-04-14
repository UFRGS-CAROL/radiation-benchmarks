INC=-I/opt/AMDAPPSDK-2.9-1/include
LIB=-L/opt/AMDAPPSDK-2.9-1/lib/x86_64

#INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
#LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

CC = gcc
CC_FLAGS = -O3

all: hot_gen hot_check

hot_gen: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_gen $(INC) OpenCL_helper_library.c hotspot.c $(LIB) -lOpenCL

hot_check: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_check $(INC) OpenCL_helper_library.c hotspot_check.c $(LIB) -lOpenCL
	
