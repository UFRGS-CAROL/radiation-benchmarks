#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string>

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

int verbose = 0;

#define INPUT_A 42.0
#define INPUT_B 42.0
#define OUTPUT_R 3.43597383680000000000e+10
#define OPS 100000000

int iterations = 100000000; // global loop iteracion

size_t r_size = 0;

typedef float tested_type;

//================== Host and device matrix ptr's
tested_type *R[3];
tested_type *d_R[3];
//====================================

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if(error == cudaSuccess) {
		return;
	} 
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.", cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit(EXIT_FAILURE);
}

cudaDeviceProp GetDevice(){
//================== Retrieve and set the default CUDA device
    cudaDeviceProp prop;
	int count=0;
	printf("Get device:");
    checkFrameworkErrors( cudaGetDeviceCount(&count) );
    for (int i=0; i< count; i++) {
        checkFrameworkErrors( cudaGetDeviceProperties( &prop, i ));
        printf( "Name: %s\n", prop.name );
    }
    int *ndevice; int dev = 0;
    ndevice = &dev;
    checkFrameworkErrors( cudaGetDevice(ndevice) );

    checkFrameworkErrors( cudaSetDevice(0) );
	checkFrameworkErrors( cudaGetDeviceProperties( &prop, 0 ) );
	printf("\ndevice: %d %s\n", *ndevice, prop.name);
	return prop;
}

double mysecond()
{
   struct timeval tp;
   struct timezone tzp;
   int i = gettimeofday(&tp,&tzp);
   return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void* safe_cudaMalloc(size_t size) {
	void* devicePtr;
	void* goldPtr;
	void* outputPtr;

	// First, alloc DEVICE proposed memory and HOST memory for device memory checking
	checkFrameworkErrors( cudaMalloc(&devicePtr, size) );
	outputPtr = malloc(size);
	goldPtr = malloc(size);
	if ((outputPtr == NULL) || (goldPtr == NULL)) {
		log_error_detail((char *)"error host malloc"); end_log_file();
		printf("error host malloc\n");
		exit(EXIT_FAILURE);
	}
	
	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	checkFrameworkErrors( cudaMemset(devicePtr, 0xAA, size) );
	memset(goldPtr, 0xAA, size);
	
	checkFrameworkErrors( cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost) );
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		checkFrameworkErrors( cudaFree(devicePtr) );
		return newDevicePtr;
	}
	// ===> END FIRST PHASE
	
	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	checkFrameworkErrors( cudaMemset(devicePtr, 0x55, size) );
	memset(goldPtr, 0x55, size);
	
	checkFrameworkErrors( cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost) );
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		checkFrameworkErrors( cudaFree(devicePtr) );
		return newDevicePtr;
	}
	// ===> END SECOND PHASE

	free(outputPtr);
	free(goldPtr);
	return devicePtr;
}

void allocCudaMemory()
{
	d_R[0] = (tested_type*) safe_cudaMalloc( r_size * sizeof( tested_type ) );
	d_R[1] = (tested_type*) safe_cudaMalloc( r_size * sizeof( tested_type ) );
	d_R[2] = (tested_type*) safe_cudaMalloc( r_size * sizeof( tested_type ) );
}

void freeCudaMemory() {
	checkFrameworkErrors( cudaFree( d_R[0] ) );
	checkFrameworkErrors( cudaFree( d_R[1] ) );
	checkFrameworkErrors( cudaFree( d_R[2] ) );
}

void setCudaMemory()
{
	checkFrameworkErrors( cudaMemset( d_R[0], 0x00, r_size * sizeof ( tested_type )) );
	checkFrameworkErrors( cudaMemset( d_R[1], 0x00, r_size * sizeof ( tested_type )) );
	checkFrameworkErrors( cudaMemset( d_R[2], 0x00, r_size * sizeof ( tested_type )) );
}

__global__ void MicroBenchmarkKernel (tested_type *d_R0, tested_type *d_R1, tested_type *d_R2)
{
	register tested_type acc = 0.0;

	#pragma unroll
	for (register unsigned int count = 0; count < OPS; count++) {
		acc = __fmaf_rn(INPUT_A, INPUT_B, acc);
	}
	
	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	d_R1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	d_R2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

void usage() {
    printf("Usage: cuda_trip_dmxm -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

void checkOutputErrors() {
	int host_errors = 0;
	int memory_errors = 0;

	#pragma omp parallel for shared(host_errors)
	for(int i=0; i<r_size; i++)
	{
		register bool checkFlag = true;
		register tested_type valOutput0 = R[0][i];
		register tested_type valOutput1 = R[1][i];
		register tested_type valOutput2 = R[2][i];
		register tested_type valOutput = valOutput0;
		if ((valOutput0 != valOutput1) || (valOutput1 != valOutput2)) {
			#pragma omp critical
			{
				char info_detail[150];
				snprintf(info_detail, 150, "m: [%d], r0: %1.20e, r1: %1.20e, r2: %1.20e", i, valOutput0, valOutput1, valOutput2);
				if (verbose && (memory_errors < 10)) printf("%s\n", info_detail);

				#ifdef LOGS
					log_info_detail(info_detail);
				#endif
				memory_errors+=1;
			}
			if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2)) {
				// All 3 values diverge
				if (valOutput0 == OUTPUT_R) {
					valOutput = valOutput0;
				} else if (valOutput1 == OUTPUT_R) {
					valOutput = valOutput1;
				} else if (valOutput2 == OUTPUT_R) {
					valOutput = valOutput2;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					checkFlag = false;
					#pragma omp critical
					{
						char error_detail[150];
						snprintf(error_detail, 150, "f: [%d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e", i, valOutput0, valOutput1, valOutput2, OUTPUT_R);
						if (verbose && (host_errors < 10)) printf("%s\n", error_detail);
		
						#ifdef LOGS
							log_error_detail(error_detail);
						#endif
						host_errors++;
					}
				}
			} else if (valOutput1 == valOutput2) {
				// Only value 0 diverge
				valOutput = valOutput1;
			} else if (valOutput0 == valOutput2) {
				// Only value 1 diverge
				valOutput = valOutput0;
			} else if (valOutput0 == valOutput1) {
				// Only value 2 diverge
				valOutput = valOutput0;
			}
		}
		// if ((fabs((double)(valOutput-valGold)/valGold) > 1e-10)||(fabs((double)(valOutput-valGold)/valGold) > 1e-10)) {
		if (OUTPUT_R != valOutput) {	
			if (checkFlag) {
				#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150, "p: [%d], r: %1.20e, e: %1.20e", i, valOutput, OUTPUT_R);
					if (verbose && (host_errors < 10)) printf("%s\n", error_detail);
	
					#ifdef LOGS
						log_error_detail(error_detail);
					#endif
					host_errors++;
				}
			}
		}
	}

	// printf("numErrors:%d", host_errors);

	if (host_errors != 0) {
		printf("#");
		#ifdef LOGS
			log_error_count(host_errors);
		#endif
		//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
		freeCudaMemory();
		//================== Init DEVICE memory
		allocCudaMemory();
		setCudaMemory();
		//====================================
	}
}

int main( int argc, char* argv[] )
{
//================== Test vars
	int loop2;
	double time;
	double kernel_time, global_time;
    double total_kernel_time, min_kernel_time, max_kernel_time;
//====================================

//================== Read test parameters
	if (checkCmdLineFlag(argc, (const char **)argv, "help") || checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
		usage();
		exit(0);
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "iterations"))
    {
        iterations = getCmdLineArgumentInt(argc, (const char **)argv, "iterations");
    }

	if (checkCmdLineFlag(argc, (const char **)argv, "verbose"))
    {
        verbose = 1;
	}
//====================================

//================== Set block and grid size for the specific device
	cudaDeviceProp prop = GetDevice();
	int gridsize = prop.multiProcessorCount;
	int blocksize = 256;
	
	printf("grid size = %d ; block size = %d\n", gridsize, blocksize);

	r_size = gridsize * blocksize;
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "ops:%d type:single-precision-triplicated", OPS);
	start_log_file((char *)"cuda_micro_sfma", test_info);
#endif
//====================================

//================== Alloc HOST memory
	R[0] = ( tested_type* ) malloc( r_size * sizeof( tested_type ) );
	R[1] = ( tested_type* ) malloc( r_size * sizeof( tested_type ) );
	R[2] = ( tested_type* ) malloc( r_size * sizeof( tested_type ) );

	if (!(R[0] && R[1] && R[2])) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
    total_kernel_time = 0;
    min_kernel_time = UINT_MAX;
    max_kernel_time = 0;
	printf( "cuda_micro_sfma\n" );
	fflush(stdout);
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	setCudaMemory();
//====================================


	for(loop2=0; loop2<iterations; loop2++)
	{
		//================== Global test loop

		global_time = mysecond();
		
		setCudaMemory();

		if (verbose) printf(",");

		kernel_time = mysecond();
		#ifdef LOGS
			start_iteration();
		#endif
		//================== Device computation, DMxM
		MicroBenchmarkKernel<<<gridsize, blocksize>>>(d_R[0], d_R[1], d_R[2]);

		checkFrameworkErrors( cudaPeekAtLastError() );
		
		checkFrameworkErrors( cudaDeviceSynchronize() );
		checkFrameworkErrors( cudaPeekAtLastError() );
		//====================================
		#ifdef LOGS
			end_iteration();
		#endif
		kernel_time = mysecond() - kernel_time;
	  
		total_kernel_time += kernel_time;
		min_kernel_time = min(min_kernel_time, kernel_time);
		max_kernel_time = max(max_kernel_time, kernel_time);

		if (verbose) printf("Device kernel time for iteration %d: %.3fs\n", loop2, kernel_time);

		//================== Gold check
    	if (verbose) printf(",");

        time = mysecond();

		checkFrameworkErrors( cudaMemcpy(R[0], d_R[0], r_size * sizeof( tested_type ), cudaMemcpyDeviceToHost) );
		checkFrameworkErrors( cudaMemcpy(R[1], d_R[1], r_size * sizeof( tested_type ), cudaMemcpyDeviceToHost) );
		checkFrameworkErrors( cudaMemcpy(R[2], d_R[2], r_size * sizeof( tested_type ), cudaMemcpyDeviceToHost) );
		checkOutputErrors();
		//====================================

		//================== Console hearthbeat
		printf(".");
		fflush(stdout);
		//====================================

		if (verbose) printf("Gold check time for iteration %d: %.3fs\n", loop2, mysecond() - time);

		if (verbose)
		{
			/////////// PERF
			double flops = r_size * OPS;
			double gflops = flops / kernel_time;
			double outputpersec = (double)r_size/kernel_time;
			printf("OPS:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", OPS, outputpersec, gflops, gflops/1000000000);
			///////////
		}

		if (verbose) printf("Iteration #%d time: %.3fs\n\n\n", loop2, mysecond() - global_time);
		fflush(stdout);
	}

    double gflops = r_size * OPS / 1000000000; // Bilion FLoating-point OPerationS
    double averageKernelTime = total_kernel_time / iterations;
    printf("\n-- END --\n"
    "Total kernel time: %.3fs\n"
    "Iterations: %d\n"
    "Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n"
    "Average GFLOPs: %.2f (best: %.2f ; worst: %.2f)\n", 
    total_kernel_time, 
    iterations, 
    averageKernelTime, min_kernel_time, max_kernel_time,
    gflops / averageKernelTime, gflops / min_kernel_time, gflops / max_kernel_time);

	//================== Release device memory
	freeCudaMemory();
	//====================================

	free( R[0] );
	free( R[1] );
	free( R[2] );
	#ifdef LOGS
		end_log_file();
	#endif

	return 0;
}