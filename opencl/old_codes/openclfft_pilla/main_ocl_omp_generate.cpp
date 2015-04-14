#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "fft.h"

#define AVOIDZERO 1e-200
#define ACCEPTDIFF 1e-5

#define TWOPI 2*M_PI
#define THREAD_PER_BLOCK 64
//#define BLOCK_SIZE 8

// 2^16 = 65535 -> size of fft_ideal.txt
// 2^20 = 1048576
// 2^25 = 33554432
#define N 1048576 // Number of Complex numbers for the FFT
//#define N 1024 // Number of Complex numbers for the FFT
#define FFTSIZE 512


int padComplex(int n, Complex* idata, Complex** pad_idata);

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage);


double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

int main(int argc, char** argv) {

	// gpu_work_percentage from 0% to 100%.
	// gpu_work_percentage default value is 100%
	int gpu_workload = 100;
	if(argc > 1) {
        gpu_workload = atoi(argv[1]);
    }

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
        printf("FFT number of points: %d\t Total size: %d\n", FFTSIZE, N);
        count_do_while++;
       
	//Preparing inputs 
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

	//setting the size of the FFT
	NFFT = padComplex(N, idata, &host_pad_idata);
	padComplex(N, odata, &host_pad_odata);

	//Calling the FFT and the IFFT
	fft_ocl_omp(NFFT, host_pad_idata, host_pad_odata, FORWARD, gpu_workload);
        fft_ocl_omp(NFFT, host_pad_odata, host_pad_idata, INVERSE, gpu_workload);
        
        for(i=0; i<N; i++) {
                host_pad_idata[i].real = host_pad_idata[i].real / FFTSIZE;
                host_pad_idata[i].imag = host_pad_idata[i].imag / FFTSIZE;
        }
        
        // inverse data check
        errors_count = 0;
        sum_zeros = 0;

        for(i = 0; i < N; i++) {
                //if(i< 20){
                        //printf("Expected: %lf\t found: %lf\t calc: %lf\n",idata[i].real, host_pad_idata[i].real, host_pad_idata[i].real*NFFT);
		//}

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

	//storing gold data
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

    } while((sum_zeros > 0 || errors_count > 0) && count_do_while < 10);
    
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

	for(i = 1; i < FFTSIZE; i <<=1) {
		log2 ++;
	}

        for(int b = 0; b < n/FFTSIZE; b++){
		for(int it = 0; it < FFTSIZE; it++) {
			int m=it;
			int is = 0;
			for(i=0; i<log2; i++) {
				int j = m/2;
				is = 2*is + (m - 2*j);
				m = j;
			}
//			odata[it] = idata[is];
			odata[b*FFTSIZE+it] = idata[b*FFTSIZE+is];

		}
	}
}

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage){

	if(gpu_work_percentage > 100 || gpu_work_percentage < 0)
        	gpu_work_percentage = 100;

	// For measuring time
	double total_time = 0;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	long long time0, time1, total_time0, total_time1;

	Cal_Unit h_unit;
	Complex *h_factor;

	bitrp(NFFT, idata, odata);

	int mem_size = sizeof(Complex)*NFFT;
	int fac_size = sizeof(Complex)*FFTSIZE/2;
	h_factor = new Complex[FFTSIZE/2];

	double delta = (direction*TWOPI)/FFTSIZE;

	for(int i = 0; i < FFTSIZE/2; i++)
	{
		double theta = i*delta;
		h_factor[i].real = cos(theta);
		h_factor[i].imag = -sin(theta);
	}

    	printf("GPU will execute several stages now\n");

	total_time0 = get_time();

	int threads = NFFT/FFTSIZE;
	int workgroup_size = THREAD_PER_BLOCK;
	//int threads = 2; //NFFT/FFTSIZE;
	//int workgroup_size = 1;//THREAD_PER_BLOCK;
	assert(threads > 0); 
	assert(workgroup_size > 0);
	assert(NFFT == FFTSIZE*threads);

	//starting OpenCL part
	initOpenCL();
	
	ocl_alloc_buffers(mem_size);

	ocl_write_odata_buffer(odata, mem_size);
	ocl_write_factor_buffer(h_factor, fac_size);

	ocl_set_kernel_args();

	time0 = get_time();

	h_unit.stage = 1;
	h_unit.unit_size = 1;
	h_unit.n = FFTSIZE;

	ocl_write_unit_buffer(&h_unit, sizeof(Cal_Unit));

	printf("Total threads = %d\t WG size = %d Total size = %d\n", threads, workgroup_size, threads*FFTSIZE);
	ocl_exec_kernel(threads, workgroup_size);

	time1 = get_time();
	gpu_kernel_time = (double) (time1-time0) / 1000000;
	
	ocl_read_odata_buffer(odata, mem_size);
	ocl_release_buffers();
	
	//ending OpenCL part
	deinitOpenCL();

	delete[] h_factor;

	total_time1 = get_time();
	total_time = (double) (total_time1 - total_time0) / 1000000;
	
	printf("\ntotal GPU time: %.12f", gpu_kernel_time);
	//printf("\ntotal CPU time: %.12f", cpu_kernel_time);
	//printf("\ntotal kernels time: %.12f", gpu_kernel_time+cpu_kernel_time);
	printf("\ntotal fft time: %.12f\n", total_time);

}
