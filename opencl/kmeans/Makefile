CC = g++
CC_FLAGS = -g -O2 

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/



SOURCES=$(wildcard *.c)


amd: $(SOURCES)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL $(SOURCES) -o kmeans $(AMD_INC) $(AMD_LIB)

intel: $(SOURCES)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL $(SOURCES) -o kmeans $(INTEL_INC) $(INTEL_LIB)

nvidia: $(SOURCES)
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL $(SOURCES) -o kmeans $(NVIDIA_INC) $(NVIDIA_LIB)

clean:
	rm -f *.o *~ kmeans *.linkinfo
