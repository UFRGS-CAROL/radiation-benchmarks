#ifndef _FFT_H
#define _FFT_H

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define FORWARD 1
#define INVERSE -1

typedef struct c{
	double real;
	double imag;
//	int stage;
} Complex;

typedef struct{
	int n;
	int stage;
	int unit_size;
}Cal_Unit;

// Returns the current system time in microseconds
inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

void fft_omp_kernel(Cal_Unit unit, Complex* factor, Complex* odata, int threads);

void initOpenCL();
void deinitOpenCL();
void ocl_alloc_buffers(int mem_size);
void ocl_release_buffers();
void ocl_set_kernel_args();
void ocl_write_odata_buffer(Complex *odata, int mem_size);
void ocl_read_odata_buffer(Complex *odata, int mem_size);
void ocl_write_factor_buffer(Complex *h_factor, int mem_size);
void ocl_write_unit_buffer(Cal_Unit *h_unit, int mem_size);
void ocl_exec_kernel(const long unsigned int global_wsize, const long unsigned int local_wsize);



#endif // _FFT_H
