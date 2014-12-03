#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft.h"

#define AVOIDZERO 1e-200
#define ACCEPTDIFF 1e-5

#define TWOPI 2*M_PI
#define THREAD_PER_BLOCK 64
#define BLOCK_SIZE 8

// 2^16 = 65535 -> size of fft_ideal.txt
// 2^20 = 1048576
// 2^25 = 33554432
#define N 2097152//4194304 // Number of Complex numbers for the FFT


int padComplex(int n, Complex* idata, Complex** pad_idata);

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage);

// Returns the current system time in microseconds
inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}


double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char** argv) {

	// gpu_work_percentage from 0% to 100%.
	// gpu_work_percentage default value is 100%
	int gpu_workload = 0;

	int NFFT;
	int i;

	Complex *host_pad_idata, *host_pad_odata;

	FILE *input;
	FILE *gold;

    // data in host memory
    Complex *idata, *odata;
    idata = (Complex *)malloc(N * sizeof(Complex));
    odata = (Complex *)malloc(N * sizeof(Complex));
    
    int count_do_while = 0;
    int errors_count = 0, sum_zeros = 0;
    do{
        printf("\n\tGenerating new input and output files\n");
        printf("FFT number of points: %d\n", N);
        count_do_while++;
        
        if( (input = fopen("input", "wb" )) == 0 ){
            printf( "The file input was not opened\n");
            exit(-1);
        }
        
	    for(i = 0; i < N; i++) {
		    idata[i].real = fRand(-3500, 35000);
		    idata[i].imag = fRand(-3500, 35000);
		    fwrite(&(idata[i].real), 1, sizeof(double), input);
		    fwrite(&(idata[i].imag), 1, sizeof(double), input);
	    }
	    
	    fclose(input);

	    NFFT = padComplex(N, idata, &host_pad_idata);
	    padComplex(N, odata, &host_pad_odata);
	    printf("NFFT = %d\n", NFFT);


	    fft_omp(NFFT, host_pad_idata, host_pad_odata, FORWARD, gpu_workload);

        fft_omp(NFFT, host_pad_odata, host_pad_idata, INVERSE, gpu_workload);
        
        for(i=0; i<N; i++) {
                host_pad_idata[i].real = host_pad_idata[i].real / NFFT;
                host_pad_idata[i].imag = host_pad_idata[i].imag / NFFT;
        }
        
        // inverse data check
        errors_count = 0;
        sum_zeros = 0;

        for(i = 0; i < N; i++) {
                if(fabs(host_pad_odata[i].real)<AVOIDZERO)
                    sum_zeros++;
                if(fabs(host_pad_odata[i].imag)<AVOIDZERO)
                    sum_zeros++;

                if ((fabs(idata[i].real)>AVOIDZERO)&&
        ((fabs((host_pad_idata[i].real-idata[i].real)/host_pad_idata[i].real)>ACCEPTDIFF)||
         (fabs((host_pad_idata[i].real-idata[i].real)/idata[i].real)>ACCEPTDIFF))) {
                    errors_count++;
                }

                if ((fabs(idata[i].imag)>AVOIDZERO)&&
        ((fabs((host_pad_idata[i].imag-idata[i].imag)/host_pad_idata[i].imag)>ACCEPTDIFF)||
         (fabs((host_pad_idata[i].imag-idata[i].imag)/idata[i].imag)>ACCEPTDIFF))) {
                    errors_count++;
                }
        }

        if( (gold = fopen("output", "wb" )) == 0 ){
            printf( "The file output was not opened\n");
            exit(-1);
        }
	    for(i = 0; i < N; i++) {
		    fwrite(&(host_pad_odata[i].real), 1, sizeof(double), gold);
		    fwrite(&(host_pad_odata[i].imag), 1, sizeof(double), gold);
	    }
	    fclose(gold);

	    free(host_pad_idata);
	    free(host_pad_odata);

	    printf("\n\tNumber of zeros in output: %d\n",sum_zeros);
	    printf("\tNumber of error executing inverse FFT: %d\n",errors_count);

    }while((sum_zeros > 0 || errors_count > 0) && count_do_while < 10);
    
    free(idata);
    free(odata);
    
	return 0;
}


int padComplex(int n, Complex* idata, Complex** pad_idata) {
	int NFFT;
	NFFT = (int)pow(2.0, ceil(log((double)n)/log(2.0)));

	Complex *new_data = (Complex *)malloc(NFFT*sizeof(Complex));

	memcpy(new_data, idata, sizeof(Complex)*n);

	*pad_idata = new_data;

	return NFFT;
}

void bitrp(int n,Complex* idata, Complex* odata) {
	int log2=0;
	int i;

	for(i = 1; i < n; i <<=1) {
		log2 ++;
	}

	for(int it = 0; it < n; it++) {
		int m=it;
		int is = 0;
		for(i=0; i<log2; i++) {
			int j = m/2;
			is = 2*is + (m - 2*j);
			m = j;
		}
		odata[it] = idata[is];
	}
}

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage){

        gpu_work_percentage = 0;

	// For measuring time
	double total_time = 0;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	long long time0, time1, total_time0, total_time1;

	
        Cal_Unit * h_unit = (Cal_Unit *)malloc(sizeof(Cal_Unit));
//	Cal_Unit h_unit;
	Complex *h_factor;
	int numThreads = BLOCK_SIZE * THREAD_PER_BLOCK;

	bitrp(NFFT, idata, odata);

	int mem_size = sizeof(Complex)*NFFT;
	h_factor = new Complex[NFFT/2];

	double delta = (direction*TWOPI)/NFFT;

	for(int i = 0; i < NFFT/2; i++)
	{
		double theta = i*delta;
		h_factor[i].real = cos(theta);
		h_factor[i].imag = -sin(theta);
	}

	int stages = 0;
	int gpu_stages = 0;
	int block_size = 1, thread_size = 1;

	for(int i = 1; i<NFFT; i <<= 1) {
		stages++;
	}
	gpu_stages = stages*((float)gpu_work_percentage/100);


	total_time0 = get_time();

	int i = 0;
	int threads = 0;
	for(i = 1; i <= numThreads; i<<=1);
	threads = i>>1;

	int unit_size = NFFT/threads;
	if(unit_size < 2)
	{
		unit_size = 2;
		threads = NFFT/unit_size;
	}

	// initialize thread_size and block_size.
	int threads_sum = threads;
	for(i = 1; i <= THREAD_PER_BLOCK; i <<= 1);
	thread_size = i >> 1;
	block_size = threads_sum/thread_size;


	h_unit->n = NFFT;
	int current_stage = 0;


	if(stages - gpu_stages > 0){

		time0 = get_time();

		for( ; current_stage < stages; current_stage++) {
			h_unit->stage = current_stage+1;

			int size = 1<<(current_stage+1);
			if(size > unit_size)
			{
				unit_size = size;
				threads = NFFT/unit_size;
			}

			if(threads < threads_sum)
			{
				if(block_size > 1)
				{
					block_size >>= 1;
					thread_size = threads/block_size;

				}
				else
					thread_size >>= 1;
				threads_sum = threads;
			}

			h_unit->unit_size = unit_size;

			fft_omp_kernel(h_unit, h_factor, odata, block_size*thread_size);

		}
		time1 = get_time();
		cpu_kernel_time = (double) (time1-time0) / 1000000;
	}
	delete[] h_factor;
	
	total_time1 = get_time();
	total_time = (double) (total_time1-total_time0) / 1000000;
	
	
	printf("\ntotal CPU time: %.12f", cpu_kernel_time);
	printf("\ntotal fft time: %.12f\n", total_time);

}
