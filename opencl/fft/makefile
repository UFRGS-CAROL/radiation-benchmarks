CC=g++

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

SOURCES=FFT_generate.cpp fftlib.cpp
CHK_EXEC=fft_check
GEN_EXEC=fft_gen

intel: intel_generate intel_chk_gold 

nvidia: nvidia_generate nvidia_chk_gold 

nvidia_generate: $(SOURCES)
	$(CC) $(SOURCES) $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_EXEC)

nvidia_chk_gold: $(SOURCES)
	$(CC) $(SOURCES) $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL  -o $(CHK_EXEC)

intel_generate: $(SOURCES)
	$(CC) $(SOURCES) $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_EXEC)

intel_chk_gold: $(SOURCES)
	$(CC) $(SOURCES) $(INTEL_INC) $(INTEL_LIB) -lOpenCL  -o $(CHK_EXEC)

