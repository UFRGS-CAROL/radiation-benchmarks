CC=g++

OMP=-fopenmp
nvidia:INC=-I/usr/local/cuda/include/
nvidia:LIB=-L/usr/local/cuda/lib64

intel:INC=-I/opt/intel/opencl-1.2-sdk-4.6.0.92/include/
intel:LIB=-L/opt/intel/opencl-1.2-4.5.0.8/lib64/

amd:INC=-I/opt/AMDAPPSDK-3.0-0-Beta/include/
amd:LIB=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/


SOURCES=FFT_check.cpp fftlib.cpp ../../include/log_helper.c 
SOURCES_GEN=FFT_generate.cpp fftlib.cpp

APP_NAME=fft

intel: generate fft fft_timing fft_err_inj 

nvidia: generate fft fft_timing fft_err_inj

amd: generate fft fft_timing fft_err_inj 

generate: $(SOURCES_GEN)
	$(CC) $(SOURCES_GEN) $(INC) $(LIB) -lOpenCL -o $(APP_NAME)_genGold

fft: $(SOURCES)
	$(CC) $(SOURCES) $(INC) $(LIB) $(OMP) -lOpenCL -DLOGS  -o $(APP_NAME)

fft_timing: $(SOURCES)
	$(CC) $(SOURCES) $(INC) $(LIB) $(OMP) -lOpenCL -DTIMING -o $(APP_NAME)_nologs_timing

fft_err_inj: $(SOURCES)
	$(CC) $(SOURCES) $(INC) $(LIB) $(OMP) -lOpenCL -DLOGS -DERR_INJ -o $(APP_NAME)_err_inj

clean:
	rm -f $(APP_NAME) $(APP_NAME)_genGold $(APP_NAME)_nologs_timing $(APP_NAME)_err_inj
