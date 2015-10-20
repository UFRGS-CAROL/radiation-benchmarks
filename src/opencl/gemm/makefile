CC=g++

OMP=-fopenmp

nvidia:INC=-I/usr/local/cuda/include/
nvidia:LIB=-L/usr/local/cuda/lib64

intel:INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
intel:LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/
intel:OMP=-openmp

amd:INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
amd:LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

GEN_IN_EXEC=genInputMatrices
APP_NAME=gemm

amd: generate gemm gemm_timing gemm_err_inj
nvidia: generate gemm gemm_timing gemm_err_inj
intel: generate gemm gemm_timing gemm_err_inj


generate: GEMM_generate.cpp generateMatrices.cpp
	$(CC) generateMatrices.cpp $(INC) $(LIB) -lOpenCL -o $(GEN_IN_EXEC)
	$(CC) GEMM_generate.cpp $(INC) $(LIB) -lOpenCL -o $(APP_NAME)_genGold

gemm: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INC) $(LIB) -DLOGS $(OMP) -lOpenCL -o $(APP_NAME)_check

gemm_timing: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INC) $(LIB) -DTIMING $(OMP) -lOpenCL -o $(APP_NAME)_nologs_timing

gemm_err_inj: GEMM.cpp ../../include/log_helper.c
	$(CC) GEMM.cpp ../../include/log_helper.c $(INC) $(LIB) -DLOGS -DERR_INJ $(OMP) -lOpenCL -o $(APP_NAME)_err_inj

clean:
	rm -f $(APP_NAME)_check $(APP_NAME)_err_inj $(APP_NAME)_nologs_timing $(APP_NAME)_genGold $(GEN_IN_EXEC) 

