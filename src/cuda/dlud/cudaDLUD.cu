/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

 // CAROL-RADIATION radiation benchmark implementation - <caio.b.lunardi at gmail.com> - 2018

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#ifdef LOGS
#include "log_helper.h"
#endif

#include "lud_kernel.cu"

#define DEFAULT_INPUT_SIZE 8192

int verbose = 0;
int fault_injection = 0;

int k=0; // k x k matrix size
int matrixSize=0; // = k * k matrix size
int iterations=100000000; // global loop iteracion
bool generate=false;

//================== Input paths
char *gold_matrix_path, *input_matrix_path;

FILE* f_INPUT;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
double *INPUT;
double *B;
double *GOLD;

double *d_INPUT;
double *d_OUTPUT;
//====================================

void GetDevice(){
//================== Retrieve and set the default CUDA device
    cudaDeviceProp prop;
    cudaError_t teste;
    int count=0;
    teste = cudaGetDeviceCount(&count);
	printf("\nGet Device Test: %s\n", cudaGetErrorString(teste));
    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "Name: %s\n", prop.name );
    }
    int *ndevice; int dev = 0;
    ndevice = &dev;
    cudaGetDevice(ndevice);

    cudaSetDevice(0);
       cudaGetDeviceProperties( &prop, 0 );
	printf("\ndevice: %d %s\n", *ndevice, prop.name);

}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void allocCudaMemory()
{
//================== CUDA error handlers
	cudaError_t malloc;
	const char *erro;
//====================================
	malloc = cudaMalloc( ( void** ) &d_INPUT, matrixSize * sizeof( double ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail("error input"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc( ( void** ) &d_OUTPUT, matrixSize * sizeof( double ) );
	erro = cudaGetErrorString(malloc);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail("error output"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure
}

void copyCudaMemory()
{
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================
	mcpy = cudaMemset(d_OUTPUT, 0, matrixSize * sizeof (double));
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail("error gpu output load memset"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure

	mcpy = cudaMemcpy( d_INPUT, INPUT, matrixSize * sizeof( double ), cudaMemcpyHostToDevice ); // PUSH A
	erro = cudaGetErrorString(mcpy);
	if(strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail("error gpu load input"); end_log_file();
#endif
		exit(EXIT_FAILURE);} //mem allocate failure
}

void generateInputMatrix(double *m) {
	#pragma omp parallel for
	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++)
		for (int j = 0; j < DEFAULT_INPUT_SIZE; j++)
			m[i * k + j] = (double) rand() / 32768.0;

	if (!(f_INPUT = fopen(input_matrix_path, "wb"))) {
		printf("Error: Could not open input file in wb mode. %s\n", input_matrix_path);
		exit(EXIT_FAILURE);
	} else {
		size_t ret_value = 0;
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
			ret_value = fwrite(&(m[i * DEFAULT_INPUT_SIZE]), DEFAULT_INPUT_SIZE * sizeof(double), 1, f_INPUT);
			if (ret_value != 1) {
				printf("Failure writing to input: %d\n", ret_value);
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_INPUT);
	}
}

void writeGoldToFile(double *m) {
	if (!(f_GOLD = fopen(gold_matrix_path, "wb"))) {
		printf("Error: Could not open gold file in wb mode. %s\n", gold_matrix_path);
		exit(EXIT_FAILURE);
	} else {
		size_t ret_value = 0;
		for (int i = 0; i < k; i++) {
			ret_value = fwrite(&(m[i * k]), k * sizeof(double), 1, f_GOLD);
			if (ret_value != 1) {
				printf("Failure writing to gold: %d\n", ret_value);
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_GOLD);
	}
}

void ReadMatrixFromFile(){
//================== Read inputs to HOST memory
	int i;
	if (verbose) printf("Reading matrices... ");
	double time = mysecond();
	f_INPUT = fopen(input_matrix_path,"rb");
	if (f_INPUT) {
		// open input successful
    	size_t ret_value;
		for(i=0; i<k; i++)
		{
			ret_value = fread (&(INPUT[ k * i ]), sizeof(double)*k, 1, f_INPUT);
			if (ret_value != 1) {
				printf("Bad input formatting: %lu .\n", ret_value);
				#ifdef LOGS
					log_error_detail("Bad input formatting."); end_log_file();
				#endif
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_INPUT);
	} else if (generate) {
		generateInputMatrix(INPUT);
	} else {
		printf ("Cant open matrices and -generate is false.\n");
		if (generate) {
			generateInputMatrix(INPUT);
		} else {
#ifdef LOGS
			log_error_detail("Cant open matrices"); end_log_file();
#endif
			exit(EXIT_FAILURE);
		}
	}

	if (!generate) {
    	size_t ret_value;
		f_GOLD = fopen(gold_matrix_path,"rb");
		for(i=0; i<k; i++)
		{
			ret_value = fread (&(GOLD[ k * i ]), sizeof(double)*k, 1, f_GOLD);
			if (ret_value != 1) {
				printf("Bad gold formatting: %lu .\n", ret_value);
				#ifdef LOGS
					log_error_detail("Bad gold formatting."); end_log_file();
				#endif
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_GOLD);
	}
	if (verbose) printf("Done reading matrices in %.2fs\n", mysecond() - time);

	if (fault_injection)
	{
		INPUT[3] = (double)6.5;
		printf("!! Injected 6.5 on position INPUT[3]\n");
	}
}

bool badass_memcmp(double *gold, double *found, unsigned long n){
	double result = 0.0;
	int i;
	unsigned long  chunk = ceil(float(n) / float(omp_get_max_threads()));
	// printf("size %d max threads %d chunk %d\n", n, omp_get_max_threads(), chunk);
	double time = mysecond();
#pragma omp parallel for default(shared) private(i) schedule(static,chunk) reduction(+:result)
   for (i=0; i < n; i++)
     result = result + (gold[i] - found[i]);

    //  printf("comparing took %lf seconds, diff %lf\n", mysecond() - time, result);
	if (fabs(result) > 0.0000000001)
		return true;
	return false;
}

void usage() {
    printf("Usage: dlud -size=N [-generate] [-input=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

int main( int argc, char* argv[] )
{
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================

//================== Test vars
	int i, j, loop2;
	// int kernel_errors=0;
	// int zero = 0;
	double time;
	double kernel_time, global_time;
    double total_kernel_time, min_kernel_time, max_kernel_time;
	int device_warmup = 1;
    // int gpu_check = 1;
//====================================

//================== Read test parameters
	if (argc<2) {
		usage();
		exit (-1);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "size"))
    {
        k = getCmdLineArgumentInt(argc, (const char **)argv, "size");

        if ((k <= 0)||(k % 16 != 0))
        {
            printf("Invalid input size given on the command-line: %d\n", k);
            exit(EXIT_FAILURE);
		}
		matrixSize = k * k;
    }
	else
	{
		usage();
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "input_a", &input_matrix_path);
    }
    else
    {
        input_matrix_path = new char[100];
        snprintf(input_matrix_path, 100, "dlud_input_%i.matrix", (signed int)DEFAULT_INPUT_SIZE);
        printf("Using default input path: %s\n", input_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "gold"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "gold", &gold_matrix_path);
    }
    else
    {
        gold_matrix_path = new char[100];
        snprintf(gold_matrix_path, 100, "dlud_gold_%i.matrix", (signed int)k);
        printf("Using default gold path: %s\n", gold_matrix_path);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))
    {
        iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        verbose = 1;
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "debug"))
    {
		fault_injection = 1;
        printf("!! Will be injected an input error\n");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "no-warmup"))
    {
		device_warmup = 0;
        printf("!! The first iteration may not reflect real timing information\n");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "generate"))
    {
		generate = 1;
		device_warmup = 0;
		iterations = 1;
		printf("Will generate input if needed and GOLD.\nIterations setted to 1. no-warmup setted to false.\n");
    } else {
		generate = 0;
	}

	// if (checkCmdLineFlag(argc, (const char **)argv, "no-gpu-gold-check"))
    // {
	// 	gpu_check = 0;
    // } else {
    //     printf("!! The gold check will happen on the GPU and fall back to CPU in case of errors\n");
    // }
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d type:double-precision", k);
	if (!generate) start_log_file("cudaDLUD", test_info);
#endif
//====================================

//================== Alloc HOST memory
	INPUT = ( double* ) malloc( matrixSize * sizeof( double ) );

	GOLD = ( double* ) malloc( matrixSize * sizeof( double ) );

	if (!(INPUT && GOLD)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
    total_kernel_time = 0;
    min_kernel_time = UINT_MAX;
    max_kernel_time = 0;
	GetDevice();
	ReadMatrixFromFile();
	printf( "cudaDLUD\n" );
	fflush(stdout);
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory();
//====================================


	for(loop2=0; loop2<iterations; loop2++)
	{//================== Global test loop

		if (!loop2 && device_warmup) printf("First iteration: device warmup. Please wait...\n");

		// Timer...
		global_time = mysecond();

		cudaMemset(d_OUTPUT, 0, matrixSize * sizeof (double));

		if (verbose) printf(",");

		kernel_time = mysecond();
		#ifdef LOGS
		if (loop2 || !device_warmup)
			if (!generate) start_iteration();
		#endif
		//================== Device computation, HMxM
		lud_cuda(d_INPUT, k);

		checkCudaErrors( cudaPeekAtLastError() );
		
		checkCudaErrors( cudaDeviceSynchronize() );
		checkCudaErrors( cudaPeekAtLastError() );
		//====================================
		#ifdef LOGS
		if (loop2 || !device_warmup)
			if (!generate) end_iteration();
		#endif
		kernel_time = mysecond() - kernel_time;
      
		if (loop2 || !device_warmup) {
		  total_kernel_time += kernel_time;
		  min_kernel_time = min(min_kernel_time, kernel_time);
		  max_kernel_time = max(max_kernel_time, kernel_time);
		}

		if (loop2 || !device_warmup)
			if (verbose) printf("Device kernel time for iteration %d: %.3fs\n", loop2, kernel_time);

    	if (verbose) printf(",");

        // Timer...
        time = mysecond();

        //if (kernel_errors != 0) {
        checkCudaErrors( cudaMemcpy(INPUT, d_OUTPUT, matrixSize * sizeof( double ), cudaMemcpyDeviceToHost) );
		if (generate) {
			writeGoldToFile(INPUT);
		} else if (loop2 || !device_warmup) {
            //~ if (memcmp(A, GOLD, sizeof(double) * k*k)) {
            if (badass_memcmp(GOLD, INPUT, matrixSize)) {
    			char error_detail[150];
    			int host_errors = 0;

                printf("!");

    			#pragma omp parallel for
    			for(i=0; (i<k); i++)
    			{
    				for(j=0; (j<k); j++)
    				{
    					if (INPUT[i + k * j] != GOLD[i + k * j])
    					#pragma omp critical
    					{
    						snprintf(error_detail, 150, "p: [%d, %d], r: %1.16e, e: %1.16e", i, j, (float)(INPUT[i + k * j]), (float)(GOLD[i + k * j]));
    						if (verbose && (host_errors < 10)) printf("%s\n", error_detail);
    						#ifdef LOGS
								if (!generate) log_error_detail(error_detail);
    						#endif
    						host_errors++;
    						//ea++;
    						//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + k * j], GOLD[i + k * j], t_ea);

    					}
    				}
    			}

                // printf("numErrors:%d", host_errors);

    			#ifdef LOGS
					if (!generate) log_error_count(host_errors);
    			#endif
    			//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
    			cudaFree( d_INPUT );
    			cudaFree( d_OUTPUT );
    			//====================================
    			ReadMatrixFromFile();
    			//================== Init DEVICE memory
    			allocCudaMemory();
    			copyCudaMemory();
    			//====================================
    		}
        }

		//====================================

		//================== Console hearthbeat
		/*if(kernel_errors > 0 || (loop2 % 10 == 0))
		{
			printf("test number: %d\n", loop2);
			printf(" kernel time: %f\n", kernel_time);
		}
		else
		{*/
			printf(".");
			fflush(stdout);
		//}
		//====================================

		if (loop2 || !device_warmup)
			if (verbose) printf("Gold check time for iteration %d: %.3fs\n", loop2, mysecond() - time);

		if (loop2 || !device_warmup)
			if (verbose)
			{
				/////////// PERF
				double outputpersec = (double)matrixSize/kernel_time;
				printf("SIZE:%d OUTPUT/S:%f\n",k, outputpersec);
				///////////
			}

		if (loop2 || !device_warmup)
			if (verbose) printf("Iteration #%d time: %.3fs\n\n\n", loop2, mysecond() - global_time);
		fflush(stdout);
	}

    double averageKernelTime = total_kernel_time / (iterations - (device_warmup ? 1 : 0));
    printf("\n-- END --\n"
    "Total kernel time: %.3fs\n"
    "Iterations: %d\n"
    "Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n", 
    total_kernel_time, 
    iterations, 
    averageKernelTime, min_kernel_time, max_kernel_time);

	//================== Release device memory
	cudaFree( d_INPUT );
	cudaFree( d_OUTPUT );
	//====================================

	free( INPUT );
	free( GOLD );
	#ifdef LOGS
		if (!generate) end_log_file();
	#endif

	return 0;
}