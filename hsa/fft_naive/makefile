all: brig fft fft_gen

brig: fft_kernel.cl
    cloc fft_kernel.cl

fft:  main_ocl_omp.c fft_omp.c fft.h
	gcc main_ocl_omp.c fft_omp.c -fopenmp hsa_helper.c elf_utils.c -lelf -lhsa-runtime64 -lm -o fft_hsa

fft_gen:  main_ocl_omp_generate.c fft_omp.c fft.h
	g++ main_ocl_omp_generate.c fft_omp.c -fopenmp -o fft_gen
    
