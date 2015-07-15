CC=gcc

NVIDIA_INC=-I/usr/local/cuda/include/
NVIDIA_LIB=-L/usr/local/cuda/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

SOURCES_GEN=main.c  ./util/opencl/opencl.c
SOURCES_CHECK=main_check.c  ./util/opencl/opencl.c ../../include/log_helper.c


amd: amd_gen amd_check amd_check_nologs

amd_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(AMD_LIB) $(AMD_INC) -o lavamd_gen

amd_check_nologs: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm -lgomp -fopenmp -lOpenCL $(AMD_LIB) $(AMD_INC) -o lavamd_check_nologs

amd_check: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DLOGS=1 -lm -lgomp -fopenmp -lOpenCL $(AMD_LIB) $(AMD_INC) -o lavamd_check

intel: intel_gen intel_check intel_check_nologs

intel_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(INTEL_LIB) $(INTEL_INC) -o lavamd_gen

intel_check_nologs: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm -lgomp -fopenmp -lOpenCL $(INTEL_LIB) $(INTEL_INC) -o lavamd_check_nologs

intel_check: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DLOGS=1 -lm -lgomp -fopenmp -lOpenCL $(INTEL_LIB) $(INTEL_INC) -o lavamd_check

nvidia: nvidia_gen nvidia_check nvidia_check_nologs

nvidia_gen: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm -lgomp -fopenmp -lOpenCL $(NVIDIA_LIB) $(NVIDIA_INC) -o lavamd_gen

nvidia_check_nologs: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm -lgomp -fopenmp -lOpenCL $(NVIDIA_LIB) $(NVIDIA_INC) -o lavamd_check_nologs

nvidia_check: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -DLOGS=1 -lm -lgomp -fopenmp -lOpenCL $(NVIDIA_LIB) $(NVIDIA_INC) -o lavamd_check

clean:
	rm lavamd_gen input_* output_*
