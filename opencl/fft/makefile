CC=g++

NVIDIA_INC=-I/usr/local/cuda/include/
NVIDIA_LIB=-L/usr/local/cuda/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/


SOURCES=FFT_check.cpp fftlib.cpp
SOURCES_GEN=FFT_generate.cpp fftlib.cpp
CHK_EXEC=fft_check
GEN_EXEC=fft_gen

intel: intel_generate intel_chk_gold 

nvidia: nvidia_generate nvidia_chk_gold 

amd: amd_generate amd_chk_gold 

nvidia_generate: $(SOURCES)
	$(CC) $(SOURCES_GEN) $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_EXEC)

nvidia_chk_gold: $(SOURCES)
	$(CC) $(SOURCES) $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL  -o $(CHK_EXEC)

intel_generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_EXEC)

intel_chk_gold: $(SOURCES)
	$(CC) $(SOURCES) $(INTEL_INC) $(INTEL_LIB) -lOpenCL  -o $(CHK_EXEC)

amd_generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(GEN_EXEC)

amd_chk_gold: $(SOURCES)
	$(CC) $(SOURCES) $(AMD_INC) $(AMD_LIB) -lOpenCL  -o $(CHK_EXEC)

clean:
	rm $(GEN_EXEC) $(CHK_EXEC) input_fft_* output_fft_*
