CC=gcc

OMP=-fopenmp

nvidia:INC=-I/usr/local/cuda/include/
nvidia:LIB=-L/usr/local/cuda/lib64

intel:INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
intel:LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

amd:INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
amd:LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

SOURCES_GEN=main.c  ./util/opencl/opencl.c
SOURCES_CHECK=main_check.c  ./util/opencl/opencl.c ../../include/log_helper.c
APP_NAME=lavamd

amd: generate lavamd lavamd_timing lavamd_err_inj

intel: generate lavamd lavamd_timing lavamd_err_inj

nvidia: generate lavamd lavamd_timing lavamd_err_inj

generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) -lm $(OMP) -lOpenCL $(LIB) $(INC) -o $(APP_NAME)_genGold

lavamd: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -lOpenCL -DLOGS $(LIB) $(INC) -o $(APP_NAME)

lavamd_timing: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -lOpenCL -DTIMING $(LIB) $(INC) -o $(APP_NAME)_nologs_timing

lavamd_err_inj: $(SOURCES_CHECK)
	$(CC) $(SOURCES_CHECK) -lm $(OMP) -lOpenCL -DLOGS -DERR_INJ $(LIB) $(INC) -o $(APP_NAME)_err_inj

clean:
	rm -f $(APP_NAME) $(APP_NAME)_check $(APP_NAME)_err_inj $(APP_NAME)_nologs_timing $(APP_NAME)_genGold

