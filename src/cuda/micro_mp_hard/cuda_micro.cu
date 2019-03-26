#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <omp.h>
#include <random>
#include <cuda_fp16.h>

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define HALF_ROUND_STYLE 1
#define HALF_ROUND_TIES_TO_EVEN 1
#include "half.hpp"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

//#define OPS 1000000000

//===================================== DEFINE TESTED PRECISION
//FOR DMR APPROACH I NEED to use the smallest precision
//as a limit, since it is not possible to store the bigger precisions
//on smaller precisions

//If double it means that DMR will be double and float
//so the limits are the float ones
#if defined(test_precision_double)

//#define OPS_PER_THREAD_OPERATION 1
//#define INPUT_A 1.1945305291614955E+103 // 0x5555555555555555
//#define INPUT_B 3.7206620809969885E-103 // 0x2AAAAAAAAAAAAAAA
//#define OUTPUT_R 4.444444444444444 //0x4011C71C71C71C71
//const char test_precision_description[] = "double";
//typedef double tested_type;
//typedef double tested_type_host;

//#elif defined(test_precision_single)

#define OPS_PER_THREAD_OPERATION 1
#define INPUT_A 1.4660155E+13 // 0x55555555
#define INPUT_B 3.0316488E-13 // 0x2AAAAAAA
#define OUTPUT_R 4.444444 //0x408E38E3
const char test_precision_description[] = "single_and_double";
typedef float tested_type;
typedef float tested_type_host;


//If single it means that DMR will be single and half
//so the limits are the half ones
#elif defined(test_precision_single)

#define OPS_PER_THREAD_OPERATION 1
#define INPUT_A 1.066E+2 // 0x56AA
#define INPUT_B 4.166E-2 // 0x2955
#define OUTPUT_R 4.44 // 0x4471
const char test_precision_description[] = "half_and_single";
typedef half tested_type;
typedef half_float::half tested_type_host;

#endif
//=====================================================

#if defined(test_type_fma) 
const char test_type_description[] = "fma_dmr";
#elif defined(test_type_add) 
const char test_type_description[] = "add_dmr";
#elif defined(test_type_mul)
const char test_type_description[] = "mul_dmr";
#endif

//====================== benchmark+setup configuration
int verbose = 0;

size_t r_size = 0;

int iterations = 100000000; // global loop iteration
//=========================

//================== Host and device matrix ptr's
tested_type_host *R;

tested_type *d_R;
//====================================

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit(EXIT_FAILURE);
}

cudaDeviceProp GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;
	printf("Get device:");
	checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
		printf("Name: %s\n", prop.name);
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	checkFrameworkErrors(cudaGetDevice(ndevice));

	checkFrameworkErrors(cudaSetDevice(0));
	checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));
	printf("\ndevice: %d %s\n", *ndevice, prop.name);
	return prop;
}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void* safe_cudaMalloc(size_t size) {
	void* devicePtr;
	void* goldPtr;
	void* outputPtr;

	// First, alloc DEVICE proposed memory and HOST memory for device memory checking
	checkFrameworkErrors(cudaMalloc(&devicePtr, size));
	outputPtr = malloc(size);
	goldPtr = malloc(size);
	if ((outputPtr == NULL) || (goldPtr == NULL)) {
		log_error_detail((char *) "error host malloc");
		end_log_file();
		printf("error host malloc\n");
		exit(EXIT_FAILURE);
	}

	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	checkFrameworkErrors(cudaMemset(devicePtr, 0xAA, size));
	memset(goldPtr, 0xAA, size);

	checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		checkFrameworkErrors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END FIRST PHASE

	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	checkFrameworkErrors(cudaMemset(devicePtr, 0x55, size));
	memset(goldPtr, 0x55, size);

	checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		checkFrameworkErrors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END SECOND PHASE

	free(outputPtr);
	free(goldPtr);
	return devicePtr;
}

void allocCudaMemory() {
	d_R = (tested_type*) safe_cudaMalloc(r_size * sizeof(tested_type));
//	d_R[1] = (tested_type*) safe_cudaMalloc(r_size * sizeof(tested_type));
//	d_R[2] = (tested_type*) safe_cudaMalloc(r_size * sizeof(tested_type));
}

void freeCudaMemory() {
	checkFrameworkErrors(cudaFree(d_R));
//	checkFrameworkErrors(cudaFree(d_R[1]));
//	checkFrameworkErrors(cudaFree(d_R[2]));
}

void setCudaMemory() {
	checkFrameworkErrors(cudaMemset(d_R, 0x00, r_size * sizeof(tested_type)));
//	checkFrameworkErrors(cudaMemset(d_R[1], 0x00, r_size * sizeof(tested_type)));
//	checkFrameworkErrors(cudaMemset(d_R[2], 0x00, r_size * sizeof(tested_type)));
}


void usage(int argc, char* argv[]) {
	printf("Usage: %s [-iterations=N] [-verbose]\n", argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool checkOutputErrors() {
	int host_errors = 0;
//	int memory_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < r_size; i++) {
		register bool checkFlag = true;
		register tested_type_host valGold = tested_type_host(OUTPUT_R);
		register tested_type_host valOutput = R[i];
		if (valGold != valOutput) {
			if (checkFlag) {
#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150, "p: [%d], r: %1.20e, e: %1.20e",
							i, (double) valOutput, (double) valGold);
					if (verbose && (host_errors < 10))
						printf("%s\n", error_detail);
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
	return host_errors == 0;
}

int main(int argc, char* argv[]) {
//================== Test vars
	int loop2;
	double time;
	double kernel_time, global_time;
	double total_kernel_time, min_kernel_time, max_kernel_time;
//====================================

//================== Read test parameters
	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		iterations = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		verbose = 1;
	}
//====================================

//================== Set block and grid size for MxM kernel
	cudaDeviceProp prop = GetDevice();
	int gridsize = prop.multiProcessorCount;
	int blocksize = 256;

	printf("grid size = %d ; block size = %d\n", gridsize, blocksize);

	r_size = gridsize * blocksize * OPS_PER_THREAD_OPERATION;
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[250];
	char test_name[250];
	snprintf(test_info, 250, "ops:%d gridsize:%d blocksize:%d type:%s-%s-precision", OPS, gridsize, blocksize, test_type_description, test_precision_description);
	snprintf(test_name, 250, "cuda_%s_micro-%s", test_precision_description, test_type_description);
	start_log_file(test_name, test_info);
#endif
//====================================

//================== Alloc HOST memory
	R = (tested_type_host*) malloc(r_size * sizeof(tested_type));
//	R[1] = (tested_type_host*) malloc(r_size * sizeof(tested_type));
//	R[2] = (tested_type_host*) malloc(r_size * sizeof(tested_type));

	if (!(R)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
	total_kernel_time = 0;
	min_kernel_time = UINT_MAX;
	max_kernel_time = 0;
	printf("cuda_micro-%s_%s\n", test_type_description,
			test_precision_description);
	fflush(stdout);
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	setCudaMemory();
//====================================

	for (loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

		global_time = mysecond();

		setCudaMemory();

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
#if test_type_fma
		MicroBenchmarkKernel_FMA<<<gridsize, blocksize>>>(d_R);
#elif test_type_add
		MicroBenchmarkKernel_ADD<<<gridsize, blocksize>>>(d_R);
#elif test_type_mul
		MicroBenchmarkKernel_MUL<<<gridsize, blocksize>>>(d_R);
#endif

		checkFrameworkErrors(cudaPeekAtLastError());

		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());
		//====================================
#ifdef LOGS
		end_iteration();
#endif
		kernel_time = mysecond() - kernel_time;

		total_kernel_time += kernel_time;
		min_kernel_time = min(min_kernel_time, kernel_time);
		max_kernel_time = max(max_kernel_time, kernel_time);

		if (verbose)
			printf("Device kernel time for iteration %d: %.3fs\n", loop2,
					kernel_time);

		//================== Gold check
		if (verbose)
			printf(",");

		time = mysecond();

		checkFrameworkErrors(
				cudaMemcpy(R, d_R, r_size * sizeof(tested_type),
						cudaMemcpyDeviceToHost));

//		checkFrameworkErrors(
//				cudaMemcpy(R[1], d_R[1], r_size * sizeof(tested_type),
//						cudaMemcpyDeviceToHost));
//
//		checkFrameworkErrors(
//				cudaMemcpy(R[2], d_R[2], r_size * sizeof(tested_type),
//						cudaMemcpyDeviceToHost));

		checkOutputErrors();
		//====================================

		//================== Console hearthbeat
		printf(".");
		fflush(stdout);
		//====================================

		if (verbose)
			printf("Gold check time for iteration %d: %.3fs\n", loop2,
					mysecond() - time);

		if (verbose) {
			/////////// PERF
			double flops = r_size * OPS * OPS_PER_THREAD_OPERATION;
			double gflops = flops / kernel_time;
			double outputpersec = (double) r_size / kernel_time;
			printf("SIZE:%ld OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", r_size,
					outputpersec, gflops, gflops / 1000000000);
			///////////
		}

		if (verbose)
			printf("Iteration #%d time: %.3fs\n\n\n", loop2,
					mysecond() - global_time);
		fflush(stdout);
	}

	double gflops = r_size * OPS * OPS_PER_THREAD_OPERATION / 1000000000; // Bilion FLoating-point OPerationS
	double averageKernelTime = total_kernel_time / iterations;
	printf("\n-- END --\n"
			"Total kernel time: %.3fs\n"
			"Iterations: %d\n"
			"Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n"
			"Average GFLOPs: %.2f (best: %.2f ; worst: %.2f)\n",
			total_kernel_time, iterations, averageKernelTime, min_kernel_time,
			max_kernel_time, gflops / averageKernelTime,
			gflops / min_kernel_time, gflops / max_kernel_time);

	//================== Release device memory
	freeCudaMemory();
	//====================================

	free(R);
//	free(R[1]);
//	free(R[2]);
#ifdef LOGS
	end_log_file();
#endif

	return 0;
}
