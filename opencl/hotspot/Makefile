CC=g++

NVIDIA_INC=-I/usr/local/cuda/include/
NVIDIA_LIB=-L/usr/local/cuda/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

CC = gcc
CC_FLAGS = -O3

nvidia: nvidia_generate nvidia_check

nvidia_generate: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_gen $(NVIDIA_INC) OpenCL_helper_library.c hotspot.c $(NVIDIA_LIB) -lOpenCL

nvidia_check: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_check $(NVIDIA_INC) OpenCL_helper_library.c hotspot_check.c $(NVIDIA_LIB) -lOpenCL
	
intel: intel_generate intel_check

intel_generate: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_gen $(INTEL_INC) OpenCL_helper_library.c hotspot.c $(INTEL_LIB) -lOpenCL

intel_check: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_check $(INTEL_INC) OpenCL_helper_library.c hotspot_check.c $(INTEL_LIB) -lOpenCL
	
amd: amd_generate amd_check

amd_generate: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_gen $(AMD_INC) OpenCL_helper_library.c hotspot.c $(AMD_LIB) -lOpenCL

amd_check: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -o hotspot_check $(AMD_INC) OpenCL_helper_library.c hotspot_check.c $(AMD_LIB) -lOpenCL
	
