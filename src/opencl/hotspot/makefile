CC=g++

nvidia:INC=-I/usr/local/cuda/include/
nvidia:LIB=-L/usr/local/cuda/lib64

intel:INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
intel:LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

amd:INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
amd:LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/

CC = gcc
CC_FLAGS = -O3
APP_NAME=hotspot

nvidia: generate hotspot hotspot_err_inj hotspot_timing

intel: generate hotspot hotspot_err_inj hotspot_timing

amd: generate hotspot hotspot_err_inj hotspot_timing

generate: hotspot.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(CC_FLAGS) $(INC) OpenCL_helper_library.c hotspot.c $(LIB) -lOpenCL -o $(APP_NAME)_genGold

hotspot: hotspot_check.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h ../../include/log_helper.c
	$(CC) $(CC_FLAGS) $(INC) ../../include/log_helper.c OpenCL_helper_library.c hotspot_check.c $(LIB) -DLOGS -lOpenCL -o $(APP_NAME)

hotspot_err_inj: hotspot_check.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h ../../include/log_helper.c
	$(CC) $(CC_FLAGS) $(INC) ../../include/log_helper.c OpenCL_helper_library.c hotspot_check.c $(LIB) -DLOGS -DERR_INJ -lOpenCL -o $(APP_NAME)_err_inj

hotspot_timing: hotspot_check.c hotspot.h OpenCL_helper_library.c OpenCL_helper_library.h
	$(CC) $(CC_FLAGS) $(INC) OpenCL_helper_library.c hotspot_check.c $(LIB) -DTIMING -lOpenCL -o $(APP_NAME)_nologs_timing

clean:
	rm -f $(APP_NAME) $(APP_NAME)_genGold $(APP_NAME)_err_inj $(APP_NAME)_nologs_timing
