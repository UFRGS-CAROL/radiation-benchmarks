#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fft.h"

#include "hsa_helper.h"

#define AVOIDZERO 1e-200
#define ACCEPTDIFF 1e-5

#define TWOPI 2*M_PI
#define THREAD_PER_BLOCK 64
#define BLOCK_SIZE 8

// 2^16 = 65535 -> size of fft_ideal.txt
// 2^20 = 1048576
// 2^25 = 33554432
#define N 2097152//4194304 // Number of Complex numbers for the FFT

#define KERNEL_NAME "&__OpenCL_fft_unit_kernel"
#define BRIG_NAME "fft_kernel.brig"

int padComplex(int n, Complex* idata, Complex** pad_idata);

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage);

// Returns the current system time in microseconds
inline long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}



int main(int argc, char** argv) {

	// gpu_work_percentage from 0% to 100%.
	// gpu_work_percentage default value is 100%
	int gpu_workload = 0;
	if(argc > 1) {
        gpu_workload = atoi(argv[1]);
    }

	int NFFT;
	int i;

	int errors_count_r, errors_count_i;

	Complex *host_pad_idata, *host_pad_odata;

	FILE *ideal_values;


    FILE *fp, *fp_gold;
    if( (fp = fopen("input", "rb" )) == 0 ) {
        printf( "error file input was not opened\n");
        exit(1);
    }
    if( (fp_gold = fopen("output", "rb" )) == 0 ) {
        printf( "error file output was not opened\n");
        exit(1);
    }

	Complex *gold = (Complex *)malloc(N*sizeof(Complex));
	Complex *idata = (Complex *)malloc(N * sizeof(Complex));
	Complex *odata = (Complex *)malloc(N * sizeof(Complex));

    int return_value, return_value2, return_value3, return_value4;
	for(i = 0; i < N; i++){
        return_value = fread(&(idata[i].real), 1, sizeof(double), fp);
        return_value2 = fread(&(idata[i].imag), 1, sizeof(double), fp);
        return_value3 = fread(&(gold[i].real), 1, sizeof(double), fp_gold);
        return_value4 = fread(&(gold[i].imag), 1, sizeof(double), fp_gold);
        if(return_value == 0 || return_value2 == 0 || return_value3 == 0 || return_value4 == 0) {
            printf("error reading input or output\n");
            exit(1);
        }
	}

	fclose(fp);
	fclose(fp_gold);


	NFFT = padComplex(N, idata, &host_pad_idata);
	padComplex(N, odata, &host_pad_odata);
	printf("NFFT = %d\n", NFFT);


	fft_ocl_omp(NFFT, host_pad_idata, host_pad_odata, FORWARD, gpu_workload);

	//data check
	errors_count_r = errors_count_i = 0;

	for(i = 0; i < N; i++) {
        if ((fabs(gold[i].real)>AVOIDZERO)&&
        ((fabs((host_pad_odata[i].real-gold[i].real)/host_pad_odata[i].real)>ACCEPTDIFF)||
         (fabs((host_pad_odata[i].real-gold[i].real)/gold[i].real)>ACCEPTDIFF))) {
if(errors_count_r <10)
	printf("Err r(%f) e(%f)\n",host_pad_odata[i].real, gold[i].real);
		    errors_count_r++;
	    }

        if ((fabs(gold[i].imag)>AVOIDZERO)&&
        ((fabs((host_pad_odata[i].imag-gold[i].imag)/host_pad_odata[i].imag)>ACCEPTDIFF)||
         (fabs((host_pad_odata[i].imag-gold[i].imag)/gold[i].imag)>ACCEPTDIFF))) {
		    errors_count_i++;
	    }
	}


	printf("Execution completed\n");
	printf("errors: %d\n", errors_count_i+errors_count_r);

	free(idata);
	free(odata);
	free(host_pad_idata);
	free(host_pad_odata);


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

	int it;
	for(it = 0; it < n; it++) {
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
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage){

    if(gpu_work_percentage > 100 || gpu_work_percentage < 0)
        gpu_work_percentage = 100;

	// For measuring time
	double total_time = 0;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	long long time0, time1, total_time0, total_time1;

	
	Cal_Unit * h_unit = (Cal_Unit *)malloc(sizeof(Cal_Unit));
	Complex *h_factor;
	int numThreads = BLOCK_SIZE * THREAD_PER_BLOCK;

	bitrp(NFFT, idata, odata);

	int mem_size = sizeof(Complex)*NFFT;
	h_factor = (Complex*) malloc( sizeof(Complex) * (NFFT/2) );// new Complex[NFFT/2];

	double delta = (direction*TWOPI)/NFFT;

	int i;
	for(i = 0; i < NFFT/2; i++)
	{
		double theta = i*delta;
		h_factor[i].real = cos(theta);
		h_factor[i].imag = -sin(theta);
	}

	int stages = 0;
	int gpu_stages = 0;
	int block_size = 1, thread_size = 1;

	for( i = 1; i<NFFT; i <<= 1) {
		stages++;
	}
	gpu_stages = stages*((float)gpu_work_percentage/100);

	printf("GPU workload percentage: %d%%\n", gpu_work_percentage);
    printf("GPU will execute %d stages of %d total stages\n", gpu_stages, stages);
    printf("CPU will execute %d stages of %d total stages\n", stages - gpu_stages, stages);

	total_time0 = get_time();

	i = 0;
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

    if(gpu_stages > 0){

/*
		initOpenCL();
		
		ocl_alloc_buffers(mem_size);

		ocl_write_odata_buffer(odata, mem_size);
		ocl_write_factor_buffer(h_factor, mem_size/2);

		ocl_set_kernel_args();
*/
		init_hsa(KERNEL_NAME, BRIG_NAME);
		set_args(h_unit, sizeof(Cal_Unit), h_factor, mem_size/2, odata, mem_size);

		time0 = get_time();

		for(current_stage = 0; current_stage < gpu_stages; current_stage++) {

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

			set_kernel_dim(thread_size, block_size*thread_size);
//			ocl_write_unit_buffer(&h_unit, sizeof(Cal_Unit));


						
//			ocl_exec_kernel(block_size*thread_size, thread_size);
			run_kernel();
//			runOclKernelHSA("fft_kernel.cl", &idata, &odata, N*sizeof(Complex), block_size*thread_size, thread_size);
		}
		time1 = get_time();
		gpu_kernel_time = (double) (time1-time0) / 1000000;
/*		
		ocl_read_odata_buffer(odata, mem_size);
		
		ocl_release_buffers();
		
		deinitOpenCL();
*/
		clean_hsa_resources();
	}

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
	free(h_factor);
	
	total_time1 = get_time();
	total_time = (double) (total_time1-total_time0) / 1000000;
	
	
	printf("\ntotal GPU time: %.12f", gpu_kernel_time);
	printf("\ntotal CPU time: %.12f", cpu_kernel_time);
	printf("\ntotal kernels time: %.12f", gpu_kernel_time+cpu_kernel_time);
	printf("\ntotal fft time: %.12f\n", total_time);

}
