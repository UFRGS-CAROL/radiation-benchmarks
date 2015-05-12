CC=gcc

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

SOURCES_GEN=main.c  ./util/opencl/opencl.c


amd: amd_gen

amd_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(AMD_LIB) $(AMD_INC) -o lavamd_gen

intel: intel_gen

intel_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(INTEL_LIB) $(INTEL_INC) -o lavamd_gen

nvidia: nvidia_gen

nvidia_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(NVIDIA_LIB) $(NVIDIA_INC) -o lavamd_gen


clean:
	rm lavamd_gen input_* output_*
