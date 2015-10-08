CC=g++

NVIDIA_INC=-I/usr/local/cuda/include/
NVIDIA_LIB=-L/usr/local/cuda/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

CHK_EXEC=gemm_check 
CHK_EXEC_NOLOGS=gemm_check_nologs 
GEN_IN_EXEC=genInputMatrices
GEN_GOLD_EXEC=genGoldMatrix

nvidia: nvidia_gemm nvidia_gen nvidia_gemm_nologs

nvidia_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(NVIDIA_INC) $(NVIDIA_LIB) -DLOGS=1 -lOpenCL -o $(CHK_EXEC)

nvidia_gemm_nologs: GEMM.cpp 
	$(CC) GEMM.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(CHK_EXEC_NOLOGS)

nvidia_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_GOLD_EXEC)
	
intel: intel_gemm intel_gen intel_gemm_nologs

intel_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INTEL_INC) $(INTEL_LIB) -DLOGS=1 -lOpenCL -o $(CHK_EXEC)

intel_gemm_nologs: GEMM.cpp
	$(CC) GEMM.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(CHK_EXEC_NOLOGS)

intel_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_GOLD_EXEC)

amd: amd_gemm amd_gen amd_gemm_nologs

amd_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(AMD_INC) $(AMD_LIB) -DLOGS=1 -lOpenCL -o $(CHK_EXEC)

amd_gemm_nologs: GEMM.cpp 
	$(CC) GEMM.cpp $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(CHK_EXEC_NOLOGS)

amd_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(GEN_GOLD_EXEC)

clean:
	rm $(CHK_EXEC) $(GEN_IN_EXEC) $(GEN_GOLD_EXEC)
