CXX = g++

ALL: generate chk_gold 

generate: FFT_generate.cpp fftlib.cpp
	g++ FFT_generate.cpp fftlib.cpp -lOpenCL -o fft_gen

chk_gold: FFT_check.cpp fftlib.cpp
	g++ FFT_check.cpp fftlib.cpp -lOpenCL -o fft_check


