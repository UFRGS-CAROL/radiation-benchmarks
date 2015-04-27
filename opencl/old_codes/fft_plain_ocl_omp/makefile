all: fft fft_gen

fft:  main_ocl_omp.cpp fft_omp.c fft_ocl.c fft.h
	g++ main_ocl_omp.cpp fft_omp.c fft_ocl.c -lOpenCL -fopenmp -o fft

fft_gen:  main_ocl_omp_generate.cpp fft_omp.c fft_ocl.c fft.h
	g++ main_ocl_omp_generate.cpp fft_omp.c fft_ocl.c -lOpenCL -fopenmp -o fft_gen
    
