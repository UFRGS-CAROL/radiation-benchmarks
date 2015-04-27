CC=g++

NVIDIA_INC=-I/usr/local/cuda-6.5/include/
NVIDIA_LIB=-L/usr/local/cuda-6.5/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

SOURCES=FFT_check.cpp fftlib.cpp
SOURCES_GEN=FFT_generate.cpp fftlib.cpp
CHK_EXEC=gemm_check
GEN_EXEC=generateMatrices

#INC=-I/usr/local/cuda-6.5/include/
#LIB=-L/usr/local/cuda-6.5/lib64


nvidia: nvidia_gemm nvidia_gen

nvidia_gemm: GEMM.cpp
	$(CC) GEMM.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(CHK_EXEC)

nvidia_gen: generateMatrices.cpp
	$(CC) generateMatrices.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_EXEC)
	
intel: intel_gemm intel_gen

intel_gemm: GEMM.cpp
	$(CC) GEMM.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(CHK_EXEC)

intel_gen: generateMatrices.cpp
	$(CC) generateMatrices.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_EXEC)
