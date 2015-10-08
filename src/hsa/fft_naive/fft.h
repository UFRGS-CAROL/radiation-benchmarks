#ifndef _FFT_H
#define _FFT_H

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#define FORWARD 1
#define INVERSE -1

typedef struct c{
	double real;
	double imag;
} Complex;

typedef struct{
	int n;
	int stage;
	int unit_size;
}Cal_Unit;


void fft_omp_kernel(Cal_Unit *unit, Complex* factor, Complex* odata, int threads);



#endif // _FFT_H
