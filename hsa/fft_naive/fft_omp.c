#include <omp.h>

#include "fft.h"

#define OMP_NUM_THREADS 8


// OpenMP single iteration of FFT
void fft_omp_kernel(Cal_Unit * unit, Complex* factor, Complex* odata, int threads) {

	omp_set_num_threads(OMP_NUM_THREADS);
	int th;
	#pragma omp parallel for private(th)
	for(th = 0; th < threads; th++){

		int threadID=th;

		int unit_start = threadID*unit->unit_size;
		int unit_end = unit_start + unit->unit_size;

		int unit_num = 1 << unit->stage;
		int jump = unit_num>>1;

		int pair;

		int i;
		for(i = unit_start; i < unit_end; i+= unit_num) {
			for(pair = i; pair <i+jump; pair++ ) {
				int match = pair + jump;

				Complex twiddle = factor[(unit->n/unit_num)*(pair%unit_num)];
				Complex temp = {twiddle.real*odata[match].real - twiddle.imag*odata[match].imag, twiddle.real*odata[match].imag + twiddle.imag*odata[match].real};

				odata[match].real = odata[pair].real - temp.real;
				odata[match].imag = odata[pair].imag - temp.imag;
				odata[pair].real = odata[pair].real + temp.real;
				odata[pair].imag = odata[pair].imag + temp.imag;
			}
		}
	}
}



