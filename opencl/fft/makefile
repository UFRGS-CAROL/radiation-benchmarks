INC=-I/usr/local/cuda-6.5/include/
LIB=-L/usr/local/cuda-6.5/lib64

ALL: generate chk_gold 

generate: FFT_generate.cpp fftlib.cpp
	g++ FFT_generate.cpp fftlib.cpp $(INC) $(LIB) -lOpenCL -o fft_gen

chk_gold: FFT_check.cpp fftlib.cpp
	g++ FFT_check.cpp fftlib.cpp $(INC) $(LIB) /home/carol/log_helper/log_helper.c -lOpenCL  -o fft_check


