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
// 2^25 = 33554432
#define N 65535 // Number of Complex numbers for the FFT


int padComplex(int n, Complex* idata, Complex** pad_idata);

// gpu_work_percentage from 0% to 100%.
// gpu_work_percentage default value is 100%
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage);



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
	errors_count_r = errors_count_i = 0;


	Complex *host_pad_idata, *host_pad_odata;

	FILE *ideal_values;


/*
---------------------------------------------------------------------
	code aimed at the golden file creation ("fft_ideal.txt");
---------------------------------------------------------------------
*/

/*
	FILE * out_ideal_values;
	out_ideal_values = fopen("fft_ideal_NEW.txt","w");
*/

/*
---------------------------------------------------------------------
	Ideal values are read by an external file and are saved in an array of Complex
---------------------------------------------------------------------
*/
	ideal_values = fopen("./fft_ideal.txt", "r");
	if (ideal_values == NULL){
		printf ("Unable to open ideal values file\n");
		exit (-1);
	}

	double real,imag;
	int temp;
	Complex *ideal_data = (Complex *)malloc(N*sizeof(Complex));
	char line[4096];

	for(i = 0; i < N; i++){
		
		fgets (line,sizeof(line),ideal_values);
		sscanf(line,"%d %lf %lf",&temp,&real,&imag);

		ideal_data[i].real = real;
		ideal_data[i].imag = imag;
		
		//printf ("%s | %f | %f\n\n", line, real,imag);
	}

	fclose (ideal_values);

		// data in host memory
		Complex *idata, *odata;
		idata = (Complex *)malloc(N * sizeof(Complex));
		odata = (Complex *)malloc(N * sizeof(Complex));

		for(i = 0; i < N; i++) {
			idata[i].real = i;
			idata[i].imag = i;
		}

		NFFT = padComplex(N, idata, &host_pad_idata);
		padComplex(N, odata, &host_pad_odata);
		printf("NFFT = %d\n", NFFT);


		fft_ocl_omp(NFFT, host_pad_idata, host_pad_odata, FORWARD, gpu_workload);

		//data check
		errors_count_r = errors_count_i = 0;

		for(i = 0; i < N; i++) {
		        if ((fabs(ideal_data[i].real)>AVOIDZERO)&&
                ((fabs((host_pad_odata[i].real-ideal_data[i].real)/host_pad_odata[i].real)>ACCEPTDIFF)||
                 (fabs((host_pad_odata[i].real-ideal_data[i].real)/ideal_data[i].real)>ACCEPTDIFF))) {
				    errors_count_r++;
				    if(errors_count_r < 10){
				        printf("[%d] Error real, e(%f) r(%f)\n", i, ideal_data[i].real, host_pad_odata[i].real);
				    }
			    }			

		        if ((fabs(ideal_data[i].imag)>AVOIDZERO)&&
                ((fabs((host_pad_odata[i].imag-ideal_data[i].imag)/host_pad_odata[i].imag)>ACCEPTDIFF)||
                 (fabs((host_pad_odata[i].imag-ideal_data[i].imag)/ideal_data[i].imag)>ACCEPTDIFF))) {
				errors_count_i++;
				if(errors_count_i < 10){
				    printf("[%d] Error imag, e(%f) r(%f)\n", i, ideal_data[i].imag, host_pad_odata[i].imag);
				}
			}
		}


		printf("Execution completed\n");
		printf("errors: %d\n", errors_count_i+errors_count_r);

		free(idata);
		free(odata);
		free(host_pad_idata);
		free(host_pad_odata);

	


/*
--------------------------------------------------------------------
	code aimed at the golden file creation ("fft_ideal.txt");
--------------------------------------------------------------------
*/
 

/*
	//printf("Transformed data:\n");
	for(i = 0; i < NFFT; i++)
	{
		fprintf(out_ideal_values,"%d %a %a\n",i,host_pad_odata[i].real,host_pad_odata[i].imag);
		host_pad_odata[i].print(i);
	}

	fclose (out_ideal_values);

*/

/*
	printf("\nTransforming data INVERSE <<<===\n");
	cuFFT(NFFT, host_pad_odata, host_pad_idata, INVERSE);


	printf("inverse data:\n");

	for(int i=0; i<N; i++)
	{
		idata[i] = host_pad_idata[i]/NFFT;
		idata[i].print(i);
	}

	for(i = 0; i < N; i++)
	{
		fprintf(output, "%lf\t%lf\n", idata[i].real, idata[i].imag);
	}

	fclose(input);
	fclose(output);

*/

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
void fft_ocl_omp(int NFFT, Complex* idata, Complex* odata,  int direction, int gpu_work_percentage){

    if(gpu_work_percentage > 100 || gpu_work_percentage < 0)
        gpu_work_percentage = 100;

	// For measuring time
	double total_time = 0;
	double gpu_kernel_time = 0, cpu_kernel_time = 0;
	long long time0, time1, total_time0, total_time1;

	
	Cal_Unit h_unit;
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

	printf("GPU workload percentage: %d%%\n", gpu_work_percentage);
    printf("GPU will execute %d stages of %d total stages\n", gpu_stages, stages);
    printf("CPU will execute %d stages of %d total stages\n", stages - gpu_stages, stages);

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


	h_unit.n = NFFT;
	int current_stage = 0;

    if(gpu_stages > 0){
		initOpenCL();
		
		ocl_alloc_buffers(mem_size);

		ocl_write_odata_buffer(odata, mem_size);
		ocl_write_factor_buffer(h_factor, mem_size/2);

		ocl_set_kernel_args();

		time0 = get_time();

		for(current_stage = 0; current_stage < gpu_stages; current_stage++) {

			h_unit.stage = current_stage+1;

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

			h_unit.unit_size = unit_size;

			ocl_write_unit_buffer(&h_unit, sizeof(Cal_Unit));

			
			ocl_exec_kernel(block_size*thread_size, thread_size);

		}
		time1 = get_time();
		gpu_kernel_time = (double) (time1-time0) / 1000000;
		
		ocl_read_odata_buffer(odata, mem_size);
		
		ocl_release_buffers();
		
		deinitOpenCL();
	}

	if(stages - gpu_stages > 0){

		time0 = get_time();

		for( ; current_stage < stages; current_stage++) {
			h_unit.stage = current_stage+1;

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

			h_unit.unit_size = unit_size;

			fft_omp_kernel(h_unit, h_factor, odata, block_size*thread_size);

		}
		time1 = get_time();
		cpu_kernel_time = (double) (time1-time0) / 1000000;
	}
	delete[] h_factor;
	
	total_time1 = get_time();
	total_time = (double) (total_time1-total_time0) / 1000000;
	
	
	printf("\ntotal GPU time: %.12f", gpu_kernel_time);
	printf("\ntotal CPU time: %.12f", cpu_kernel_time);
	printf("\ntotal kernels time: %.12f", gpu_kernel_time+cpu_kernel_time);
	printf("\ntotal fft time: %.12f\n", total_time);

}
