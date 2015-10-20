CC=g++

NVIDIA_INC=-I/usr/local/cuda/include/
NVIDIA_LIB=-L/usr/local/cuda/lib64

INTEL_INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
INTEL_LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

AMD_INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
AMD_LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

GEN_IN_EXEC=genInputMatrices
APP_NAME=gemm

nvidia: nvidia_gemm nvidia_gen nvidia_gemm_timing nvidia_gemm_err_inj

nvidia_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(NVIDIA_INC) $(NVIDIA_LIB) -DLOGS -lOpenCL -o $(APP_NAME)_check

nvidia_gemm_timing: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(NVIDIA_INC) $(NVIDIA_LIB) -DTIMING -lOpenCL -o $(APP_NAME)_nologs_timing

nvidia_gemm_err_inj: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(NVIDIA_INC) $(NVIDIA_LIB) -DLOGS -DERR_INJ -lOpenCL -o $(APP_NAME)_err_inj


nvidia_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(NVIDIA_INC) $(NVIDIA_LIB) -lOpenCL -o $(APP_NAME)_genGold

	
intel: intel_gemm intel_gen intel_gemm_timing intel_gemm_err_inj

intel_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INTEL_INC) $(INTEL_LIB) -DLOGS -lOpenCL -o $(APP_NAME)_check

intel_gemm_timing: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INTEL_INC) $(INTEL_LIB) -DTIMING -lOpenCL -o $(APP_NAME)_nologs_timing

intel_gemm_err_inj: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INTEL_INC) $(INTEL_LIB) -DLOGS -DERR_INJ -lOpenCL -o $(APP_NAME)_err_inj

intel_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(INTEL_INC) $(INTEL_LIB) -lOpenCL -o $(APP_NAME)_genGold


amd: amd_gemm amd_gen amd_gemm_timing amd_gemm_err_inj

amd_gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(AMD_INC) $(AMD_LIB) -DLOGS -lOpenCL -o $(APP_NAME)_check

amd_gemm_timing: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(AMD_INC) $(AMD_LIB) -DTIMING -lOpenCL -o $(APP_NAME)_nologs_timing

amd_gemm_err_inj: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(AMD_INC) $(AMD_LIB) -DLOGS -DERR_INJ -lOpenCL -o $(APP_NAME)_err_inj

amd_gen: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(AMD_INC) $(AMD_LIB) -lOpenCL -o $(APP_NAME)_genGold

clean:
	rm $(APP_NAME)_check $(APP_NAME)_err_inj $(APP_NAME)_nologs_timing $(GEN_IN_EXEC) 
