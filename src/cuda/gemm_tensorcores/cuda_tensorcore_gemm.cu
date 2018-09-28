/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix multiplication
// and addition: D = alpha * A * B + beta * C. The dimensions of both C and D matrices
// are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x K_GLOBAL (row-major), the B matrix
// is K_GLOBAL x N_GLOBAL (column-major).
// In that kernel, each CTA computes one 128 x 128 tile of the resulting matrix
// per iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes eight
// 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array.
// Warps compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and accumulating
// the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments from
//   shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the A and B
//   data from shared memory, thus reducing the number of data copies from global memory.
// - The portions of the A and B matrices are stored in shared memory with an additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory contents to
//   global memory, again avoiding redundant random global memory accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register utilization,
//   but carefully enough to avoid local memory use.

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <mma.h>

// helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include <helper_cuda.h>

// Externally configurable parameters.
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

#include "half.hpp"


#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

//=========== DEFINE TESTED TYPE
#if defined(test_precision_double)
	#define GENERATOR_MAXABSVALUE 4.1e+16
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "double";
	typedef double tested_type;
	typedef double tested_type_host;
#elif defined(test_precision_single)
	#define GENERATOR_MAXABSVALUE 4.1e+2
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "single";
	typedef float tested_type;
	typedef float tested_type_host;
#elif defined(test_precision_half)
	#define GENERATOR_MAXABSVALUE 2.0
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "half";
	typedef half tested_type;
	typedef half_float::half tested_type_host;
#else 
	#error TEST TYPE NOT DEFINED OR INCORRECT. USE TYPE=<double|single|half>.
#endif

//====================== benchmark+setup configuration
int generate = 0;
int verbose = 0;
int fault_injection = 0;

unsigned long long int host_is_memory_bad = 0;

int k = 0; // k x k matrix size
int matrixSize = 0; // = k * k matrix size
int iterations = 100000000; // global loop iteracion
//=========================

//======== generator configuration
int generate_safechecks = 0;
bool generate_inputmatricesready = false;
bool host_check = false;
bool generator_debug = false;
//=========================

//================== Input paths
char *gold_matrix_path, *a_matrix_path, *b_matrix_path, *c_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_C;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
tested_type_host *A;
tested_type_host *B;
tested_type_host *C;
tested_type_host *D0, *D1, *D2;
tested_type_host *GOLD;

tested_type *d_A0, *d_A1, *d_A2;
tested_type *d_B0, *d_B1, *d_B2;
tested_type *d_C0, *d_C1, *d_C2;
tested_type *d_D0, *d_D1, *d_D2;
//====================================




#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the performance
// would be severely impacted. So we choose to reduce the chunk size in half,
// i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B matrix
// in shared memory to minimize possible bank conflicts.
// Before performing the nvcuda::wmma::mma_sync operation, the warp must load the matrix
// data using the nvcuda::wmma::load_matrix_sync operation. Although the memory access pattern
// is not specified for that function, each lane in the warp can read one or multiple matrix
// elements from different matrix rows or columns.
// For shared memory, such access can result in bank conflicts if different rows / columns
// of the matrix map to the same bank. By shifting each row and column by a few bytes, we
// make sure that they map to different banks, thus reducing the number of possible bank
// conflicts.
// The number of 8 two-byte "half" elements is chosen as the minimum possible shift because
// we must keep each row and column 128-bit aligned, as required by nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)

void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	if (!generate)
		log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

void GetDevice() {
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
		exit (EXIT_FAILURE);
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

#ifdef SAFE_MALLOC
	d_A0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_A1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_A2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));

	d_B0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_B1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_B2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	
	d_C0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_C1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_C2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));

	d_D0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_D1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	d_D2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
	
#else
	checkFrameworkErrors(cudaMalloc(&d_A0, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_A1, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_A2, matrixSize * sizeof(tested_type)));

	checkFrameworkErrors(cudaMalloc(&d_B0, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_B1, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_B2, matrixSize * sizeof(tested_type)));
	
	checkFrameworkErrors(cudaMalloc(&d_C0, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_C1, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_C2, matrixSize * sizeof(tested_type)));


	checkFrameworkErrors(cudaMalloc(&d_D0, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_D1, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_D2, matrixSize * sizeof(tested_type)));
#endif

}

void freeCudaMemory() {
	checkFrameworkErrors(cudaFree(d_A0));
	checkFrameworkErrors(cudaFree(d_A1));
	checkFrameworkErrors(cudaFree(d_A2));

	checkFrameworkErrors(cudaFree(d_B0));
	checkFrameworkErrors(cudaFree(d_B1));
	checkFrameworkErrors(cudaFree(d_B2));
	
	checkFrameworkErrors(cudaFree(d_C0));
	checkFrameworkErrors(cudaFree(d_C1));
	checkFrameworkErrors(cudaFree(d_C2));

	checkFrameworkErrors(cudaFree(d_D0));
	checkFrameworkErrors(cudaFree(d_D1));
	checkFrameworkErrors(cudaFree(d_D2));
}

void copyCudaMemory() {
	checkFrameworkErrors(cudaMemset(d_D0, 0x00, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMemset(d_D1, 0x00, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMemset(d_D2, 0x00, matrixSize * sizeof(tested_type)));

	checkFrameworkErrors(
			cudaMemcpy(d_A0, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH A
	checkFrameworkErrors(
			cudaMemcpy(d_A1, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH A
	checkFrameworkErrors(
			cudaMemcpy(d_A2, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH A

	checkFrameworkErrors(
			cudaMemcpy(d_B0, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH B
	checkFrameworkErrors(
			cudaMemcpy(d_B1, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH B
	checkFrameworkErrors(
			cudaMemcpy(d_B2, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH B
					
	checkFrameworkErrors(
			cudaMemcpy(d_C0, C, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH C
	checkFrameworkErrors(
			cudaMemcpy(d_C1, C, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH C
	checkFrameworkErrors(
			cudaMemcpy(d_C2, C, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH C				
}

void readMatricesFromFile(bool gold = true) {
	int i;
	f_A = fopen(a_matrix_path, "rb");
	f_B = fopen(b_matrix_path, "rb");
	f_C = fopen(c_matrix_path, "rb");
	
	if (!(f_A && f_B && f_C)) {
		printf("Cant open input  matrices.\n");
#ifdef LOGS
		if (!generate)
			log_error_detail((char *)"Cant open input matrices"); end_log_file();
#endif
		exit(-3);
	}
	if (gold) {
 		if (! (f_GOLD = fopen(gold_matrix_path, "rb"))) {
			printf("Cant open gold matrice.\n");
#ifdef LOGS
					if (!generate)
						log_error_detail((char *)"Cant open gold matrice"); end_log_file();
#endif
			exit(-3);
		}
	}


	size_t ret_value[4];
	for (i = 0; i < k; i++) {
		ret_value[0] = fread(&(A[k * i]), sizeof(tested_type) * k, 1, f_A);
		ret_value[1] = fread(&(B[k * i]), sizeof(tested_type) * k, 1, f_B);
		ret_value[2] = fread(&(C[k * i]), sizeof(tested_type) * k, 1, f_C);
		if (gold) {
			ret_value[3] = fread(&(GOLD[k * i]), sizeof(tested_type) * k, 1, f_GOLD);
		}
		if ((ret_value[0] != 1) || (ret_value[1] != 1) || (ret_value[2] != 1) || (gold && (ret_value[3] != 1))) {
			printf("Bad input/gold formatting: %lu ; %lu ; %lu ; %lu .\n",
					ret_value[0], ret_value[1], ret_value[2], ret_value[3]);
#ifdef LOGS
			if (!generate)
				log_error_detail((char *)"Bad input/gold formatting."); end_log_file();
#endif
			exit(-3);
		}
	}

	fclose(f_A);
	fclose(f_B);
	fclose(f_C);
	if (gold) fclose(f_GOLD);
}

void generateInputMatrices()
{
	FILE *f_A, *f_B, *f_C;
	tested_type_host *h_A, *h_B, *h_C;

	if (k==DEFAULT_INPUT_SIZE) {
		h_A = A;
		h_B = B;
		h_C = C;
	} else {
		h_A = (tested_type_host*) malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		h_B = (tested_type_host*) malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		h_C = (tested_type_host*) malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		if (!(h_A && h_B && h_C)) {
			printf("Could not alloc h_A or h_B or h_C");
			exit(EXIT_FAILURE);
		}
	}

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(-GENERATOR_MAXABSVALUE, GENERATOR_MAXABSVALUE);

	if (!generator_debug) {
		for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
			for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)dis(gen);
				h_B[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)dis(gen);
				h_C[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)dis(gen);
			}
		}
	} else {
		for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
			for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)2.0;
				h_B[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)2.0;
				h_C[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)2.0;
			}
		}
	}

	if (h_A != A) {
		memcpy(A, h_A, matrixSize * sizeof(tested_type));
		memcpy(B, h_B, matrixSize * sizeof(tested_type));
		memcpy(C, h_C, matrixSize * sizeof(tested_type));
	} 

	int numZeros;
    int numNans;
    int numInfs;
// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");
	f_C = fopen(c_matrix_path, "wb");
	if (!(f_A&&f_B && f_C)) {
		printf("Could not open f_A or f_B or f_B \n");
		exit(EXIT_FAILURE);
	}

    tested_type_host val;

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=h_A[i];
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix A: %d/%d/%d\n", numZeros, numNans, numInfs);

	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=h_B[i];
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix B: %d/%d/%d\n", numZeros, numNans, numInfs);
	
	numZeros = 0;
    numNans = 0;
    numInfs = 0;
	for (int i = 0; i<DEFAULT_INPUT_SIZE*DEFAULT_INPUT_SIZE; i++) {
        val=h_C[i];
		if (val == 0) numZeros++;
        if (isnan(val)) numNans++;
        if (isinf(val)) numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix C: %d/%d/%d\n", numZeros, numNans, numInfs);

	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]), sizeof(tested_type) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f\n", (double)A[32]);
	printf("Element 50 of matrix B: %f\n", (double)B[50]);
	printf("Element 50 of matrix C: %f\n", (double)C[50]); 
	
	//printf("\nteste");
	


	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]), sizeof(tested_type_host) * DEFAULT_INPUT_SIZE, 1, f_B);
	}
	
	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_C[i * DEFAULT_INPUT_SIZE]), sizeof(tested_type_host) * DEFAULT_INPUT_SIZE, 1, f_C);
	}
	
	printf("Done\n");

	fclose(f_A);
	fclose(f_B);
	fclose(f_C);
	if (h_A != A) {
		free(h_A);
		free(h_B);
		free(h_C);
	}
	return;
}

void retrieveInputMatrices() {
//================== Read inputs to HOST memory
	double time = mysecond();

	if (verbose)
		printf("Preparing input matrices... ");

	FILE *f_A = fopen(a_matrix_path, "rb");
	FILE *f_B = fopen(b_matrix_path, "rb");
	FILE *f_C = fopen(c_matrix_path, "rb");
	if (generate && (!f_A || !f_B || !f_C)) {
		if (f_A) fclose(f_A);
		if (f_B) fclose(f_B);
		if (f_C) fclose(f_C);
		generateInputMatrices();
	} else {
		if (f_A) fclose(f_A);
		if (f_B) fclose(f_B);
		if (f_C) fclose(f_C);
		readMatricesFromFile(!generate);
	}

	if ((generate) && (generator_debug) && (k <= 16)) {
		printf("\nMatrix A: \n");
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (float)A[i]);
			if ((i+1)%k == 0) printf("\n");
		}
		printf("\nMatrix B: \n");
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (float)B[i]);
			if ((i+1)%k == 0) printf("\n");
		}
		printf("\nMatrix C: \n");
		for (int i = 0; i<k*k; i++) {
			printf(" %.2e", (float)C [i]);
			if ((i+1)%k == 0) printf("\n");
		}
	
	}

	if (fault_injection) {
		A[3] = (tested_type_host) 1.666;
		printf("!! Injected 1.666 on position A[3]\n");
	}
	
	if (verbose)
		printf("Done reading matrices in %.2fs\n", mysecond() - time);
}

void writeGoldtoFile() {
	int i;
	f_GOLD = fopen(gold_matrix_path, "wb");
	if (!f_GOLD) {
		printf("Could not open f_GOLD\n");
		exit(EXIT_FAILURE);
	}

	for(i=0; i<k; i++)
	{
		fwrite( &(GOLD[i * k]), sizeof(tested_type)*k, 1, f_GOLD );
	}

	fclose(f_GOLD);
}

__device__ unsigned long long int is_memory_bad = 0;

#if defined(test_precision_double) or defined(test_precision_single)
__device__ tested_type inline read_voter(tested_type *v1, tested_type *v2, tested_type *v3,
		int offset) {

	register tested_type in1 = v1[offset];
	register tested_type in2 = v2[offset];
	register tested_type in3 = v3[offset];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}
#elif defined(test_precision_half)
__device__ half inline read_voter(half *v1, half *v2, half *v3,
		int offset) {

	register half in1 = v1[offset];
	register half in2 = v2[offset];
	register half in3 = v3[offset];

	if (__heq(in1, in2) || __heq(in1, in3)) {
		return in1;
	}

	if (__heq(in2, in3)) {
		return in2;
	}

	if (__hne(in1, in2) && __hne(in2, in3) && __hne(in1, in3)) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}
#endif

#if defined(test_precision_half)
__device__ half2 inline read_voter_h2(half2 *v1, half2 *v2, half2 *v3,
		int offset) {

	register half2 in1 = v1[offset];
	register half2 in2 = v2[offset];
	register half2 in3 = v3[offset];

	if (__hbeq2(in1, in2) || __hbeq2(in1, in3)) {
		return in1;
	}

	if (__hbeq2(in2, in3)) {
		return in2;
	}

	if (__hbne2(in1, in2) && __hbne2(in2, in3) && __hbne2(in1, in3)) {
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}
#endif


//using namespace nvcuda;

//~ __host__ void init_host_matrices(float *a, float *b, float *c){
    //~ for (int i = 0; i < M_GLOBAL; i++) {
        //~ for (int j = 0; j < K_GLOBAL; j++) {
            //~ a[i*K_GLOBAL+j] = (float)(rand() % 3);
        //~ }
    //~ }

    //~ for (int i = 0; i < N_GLOBAL; i++) {
        //~ for (int j = 0; j < K_GLOBAL; j++) {
            //~ b[i*K_GLOBAL+j] = (float)(rand() % 3);
        //~ }
    //~ }

    //~ for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
        //~ c[t] =  (float)(rand() % 3);
    //~ }
//~ }

//~ __global__ void init_device_matrices(const float *A_h, const float *B_h, const float *C_h, half *A, half *B, float *C, float *D)
//~ {
    //~ for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * K_GLOBAL; i += gridDim.x * blockDim.x)
        //~ A[i] = __float2half(A_h[i]);

    //~ for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N_GLOBAL * K_GLOBAL; i += gridDim.x * blockDim.x)
        //~ B[i] = __float2half(B_h[i]);

    //~ for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL; i += gridDim.x * blockDim.x)
        //~ C[i] = C_h[i];

    //~ for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL; i += gridDim.x * blockDim.x)
        //~ D[i] = 0;
//~ }

//~ __global__ void compute_gemm(const half *A, const half *B, const float *C, float *D, float alpha, float beta)
//~ {
    //~ extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    //~ // Warp and lane identification.
    //~ const unsigned int warpId = threadIdx.x / WARP_SIZE;
    //~ const unsigned int laneId = threadIdx.x % WARP_SIZE;

    //~ // Offset in shared memory from which the B matrix is stored.
    //~ const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    //~ // This pointer is used to access the C and D matrix tiles this warp computes.
    //~ float *shmem_warp_tile_ptr = (float*)&shmem[0][0] + (warpId/2) * SHMEM_STRIDE * K * 2 + (warpId%2) * SHMEM_OFFSET;

    //~ // This pointer is used to stream the C and D matrices block-wide tile to and from shared memory.
    //~ float *shmem_warp_stream_ptr = (float*)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    //~ // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    //~ // each tile computation. Technically this is not generally correct (may result
    //~ // in a loss of precision). Zero still needs to be specially handled though.
    //~ beta /= alpha;

    //~ // Each CTA slides along the 128 x 128 tiles from the top left corner of the matrix to the
    //~ // right and down, and selects the next tile to compute. Once there's no such tile,
    //~ // all warps in this CTA exit.
    //~ for(unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        //~ const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        //~ const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        //~ // Stop when there are no more D matrix tiles to compute in this CTA.
        //~ if (block_tile_i >= M_TILES) {
            //~ break;
        //~ }

        //~ // This warp's pointer to the C matrix data to copy memory from to shared memory.
        //~ const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        //~ const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        //~ // Stream multiple C tiles to shared memory.
//~ #pragma unroll
        //~ for (int i = 0; i < K; i++) {
            //~ typedef int4 copy_t;

            //~ *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) = 
                //~ *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        //~ }

        //~ __syncthreads();

        //~ // These fragments will accumulate the result of A and B matrix fragment multiplications
        //~ // along the K_GLOBAL dimension.
        //~ wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        //~ // Load the C matrix tiles into fragments from shared memory.
//~ #pragma unroll
        //~ for (int i = 0; i < WARP_COL_TILES; i++) {
//~ #pragma unroll
            //~ for (int j = 0; j < WARP_ROW_TILES; j++) {
                //~ const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                //~ wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            //~ }
        //~ }

        //~ __syncthreads();

        //~ // Scale the C matrix.
//~ #pragma unroll
       //~ for (int i = 0; i < WARP_COL_TILES; i++) {
//~ #pragma unroll
            //~ for (int j = 0; j < WARP_ROW_TILES; j++) {
//~ #pragma unroll
                //~ for (int t = 0; t < c[i][j].num_elements; t++) {
                    //~ c[i][j].x[t] *= beta;
                //~ }
            //~ }
        //~ }

        //~ // Select what warp copies what matrix to shared memory.
        //~ // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        //~ const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2) :
                                              //~ (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

        //~ // Go through the global K dimension by a fixed step at a time.
//~ #pragma unroll
        //~ for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            //~ // Copy slices of the A and B matrices to shared memory.
            //~ // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            //~ size_t shmem_idx = warpId < (WARPS_PER_BLOCK/2) ? (M * (warpId % (WARPS_PER_BLOCK/2)) * 2) : 
                                                              //~ (N * (warpId % (WARPS_PER_BLOCK/2)) * 2 + shmem_idx_b_off);

            //~ // First half of the warp copies the first row / column of the matrix,
            //~ // the second half of the warp copies the next.
            //~ int4 *lane_ptr = (int4*)(warp_ptr + tile_k * K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) + (laneId % CHUNK_COPY_LINE_LANES);

            //~ // Shift the second half of the warp to the next row / column in the shared memory.
            //~ shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

//~ #pragma unroll
            //~ for(int i = 0; i < ((WARP_SIZE/2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                //~ // Copy 16 bytes at once in each lane.
                //~ *((int4*)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                //~ // Advance the global memory pointer and the shared memory index.
                //~ lane_ptr = (int4*)((half*)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                //~ shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            //~ }

            //~ __syncthreads();

            //~ // Compute a grid of C matrix tiles in each warp.
//~ #pragma unroll
            //~ for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                //~ wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
                //~ wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

//~ #pragma unroll
                //~ for (int i = 0; i < WARP_COL_TILES; i++) {
                    //~ size_t shmem_idx_a = (warpId/2) * M * 2 + (i * M);
                    //~ const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    //~ wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

//~ #pragma unroll
                    //~ for (int j = 0; j < WARP_ROW_TILES; j++) {
                        //~ if (i == 0) {
                            //~ // Load the B matrix fragment once, because it is going to be reused
                            //~ // against the other A matrix fragments.
                            //~ size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId%2) + (j * N);
                            //~ const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            //~ wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        //~ }

                        //~ wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    //~ }
                //~ }
            //~ }

            //~ __syncthreads();
        //~ }

        //~ // Store the D fragments to shared memory.
//~ #pragma unroll
        //~ for (int i = 0; i < WARP_COL_TILES; i++) {
//~ #pragma unroll
            //~ for (int j = 0; j < WARP_ROW_TILES; j++) {
//~ #pragma unroll
                //~ // Uniform, point-wise transformations of ALL fragment elements by ALL threads in the
                //~ // warp are well-defined even though element indices within fragment storage are not defined.
                //~ for (int t = 0; t < c[i][j].num_elements; t++)
                    //~ c[i][j].x[t] *= alpha;

                //~ float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                //~ wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            //~ }
        //~ }

        //~ __syncthreads();

        //~ // Now that shared memory contains all the D tiles, stream them to global memory.
        //~ float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

//~ #pragma unroll
        //~ for (int i = 0; i < K; i++) {
            //~ *((int4*)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                //~ *((int4*)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        //~ }

        //~ __syncthreads();
    //~ }
//~ }


//~ // Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//~ //  1) Matrices are packed in memory.
//~ //  2) M, N and K are multiples of 16. 
//~ //  3) Neither A nor B are transposed.
//~ // Note: This is a less performant version of the compute_gemm kernel. It is designed for
//~ //       demonstration purposes only to show the CUDA WMMA API use without relying on
//~ //       availability of the shared memory.
//~ __global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld, int n_ld, int k_ld, float alpha, float beta)
//~ {
   //~ // Leading dimensions. Packed with no transpositions.
   //~ int lda = m_ld;
   //~ int ldb = k_ld;
   //~ int ldc = n_ld;

   //~ // Tile using a 2D grid
   //~ int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   //~ int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   //~ // Declare the fragments
   //~ wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
   //~ wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   //~ wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   //~ wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   //~ wmma::fill_fragment(acc_frag, 0.0f);

   //~ // Loop over k
   //~ for (int i = 0; i < k_ld; i += WMMA_K) {
      //~ int aCol = i; 
      //~ int aRow = warpM * WMMA_M;

      //~ int bCol = i;
      //~ int bRow = warpN * WMMA_N;

      //~ // Bounds checking
      //~ if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
         //~ // Load the inputs
         //~ wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
         //~ wmma::load_matrix_sync(b_frag, b + bCol + bRow * ldb, ldb);
 
         //~ // Perform the matrix multiplication
         //~ wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      //~ }
   //~ }

   //~ // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   //~ int cCol = warpN * WMMA_N;
   //~ int cRow = warpM * WMMA_M;

   //~ if (cRow < m_ld && cCol < n_ld) {
      //~ wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc, wmma::mem_row_major);

      //~ for(int i=0; i < c_frag.num_elements; i++) {
         //~ c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      //~ }

      //~ // Store the output
      //~ wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc, wmma::mem_row_major);
   //~ }
//~ }

//~ __host__ void matMultiplyOnHost(float *A, float *B, float *C,
                                //~ float alpha, float beta,
                                //~ int numARows, int numAColumns,
                                //~ int numBRows, int numBColumns,
                                //~ int numCRows, int numCColumns)
//~ {
    //~ for (int i = 0; i < numCRows; i++) {
        //~ for (int j = 0; j < numCColumns; j++) {
            //~ float temp = 0.0;

            //~ for (int k = 0; k < numAColumns; k++) {
                //~ temp += A[i * numAColumns + k] * B[j * numBRows + k];
            //~ }

            //~ C[i*numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        //~ }
    //~ }
//~ }


  __global__ void MatrixMulKernel(tested_type *d_A0, tested_type *d_A1, tested_type *d_A2,
								tested_type *d_B0, tested_type *d_B1, tested_type *d_B2,
								tested_type *d_C0, tested_type *d_C1, tested_type *d_C2,  
								tested_type *d_D0, tested_type *d_D1, tested_type *d_D2, 
								int n) {

#if defined(test_precision_double) or defined(test_precision_single)
	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	register tested_type acc = 0.0;
	for (k = 0; k < n; k++) {
		acc = 	read_voter(d_A0, d_A1, d_A2, ty * n + k) 
				* 
				read_voter(d_B0, d_B1, d_B2, k * n + tx)
				+
				acc;
	}
	//~ acc += read_voter(d_C0, d_C1, d_C2, k * n + tx);
	acc += read_voter(d_C0, d_C1, d_C2, ty * n + tx);

	d_D0[ty * n + tx] = acc;
	d_D1[ty * n + tx] = acc;
	d_D2[ty * n + tx] = acc;

#elif defined(test_precision_half)

	register int tx = (blockIdx.x * BLOCK_SIZE) / 2.0 + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

	register half2 acc = __float2half2_rn(0.0);
	for (k = 0; k < n; k++) {

		acc = __hfma2(	__half2half2(read_voter(d_A0, d_A1, d_A2, ty * n + k)), 
						read_voter_h2((half2*)d_B0, (half2*)d_B1, (half2*)d_B2, k * (n / 2.0) + tx), 
						acc);
		// n/2 is needed because we changed how we iterate d_B
	}

	((half2*)d_C0)[ty * (n / 2) + tx] = acc;
	((half2*)d_C1)[ty * (n / 2) + tx] = acc;
	((half2*)d_C2)[ty * (n / 2) + tx] = acc;
#endif

}
void usage(int argc, char* argv[]) {
	printf("Usage: %s -size=N [-generate] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n", argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool checkOutputErrors(tested_type_host* votedOutput = NULL, bool check = true) {
	int host_errors = 0;
	int memory_errors = 0;


	if (host_is_memory_bad != 0) {
		char info_detail[150];
		snprintf(info_detail, 150,
				"b: is_memory_bad: %llu",
				host_is_memory_bad);
		if (verbose)
			printf("%s\n", info_detail);

#ifdef LOGS
		if (!generate) 
			log_info_detail(info_detail);
#endif
		memory_errors++;
	} 

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < matrixSize; i++) {
		register bool checkFlag = true;
		register tested_type_host valGold = GOLD[i];
		register tested_type_host valOutput0 = D0[i];
		register tested_type_host valOutput1 = D1[i];
		register tested_type_host valOutput2 = D2[i];
		register tested_type_host valOutput = valOutput0;
		if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
			#pragma omp critical
			{
				char info_detail[150];
				snprintf(info_detail, 150,
						"m: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e",
						(int) floor(i / k), i % k, 
						(double)valOutput0, (double)valOutput1,
						(double)valOutput2);
				if (verbose && (memory_errors < 10))
					printf("%s\n", info_detail);

#ifdef LOGS
				if (!generate) 
					log_info_detail(info_detail);
#endif
				memory_errors++;
			}
			if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2) && (valOutput0 != valOutput2)) {
				// All 3 values diverge
				if (valOutput0 == valGold) {
					valOutput = valOutput0;
				} else if (valOutput1 == valGold) {
					valOutput = valOutput1;
				} else if (valOutput2 == valGold) {
					valOutput = valOutput2;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					checkFlag = false;
					#pragma omp critical
					{
						char info_detail[150];
						snprintf(info_detail, 150,
								"t: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e",
								(int) floor(i / k), i % k, (double)valOutput0,
								(double)valOutput1, (double)valOutput2, (double)valGold);
						if (verbose && (memory_errors < 10))
							printf("%s\n", info_detail);

#ifdef LOGS
						if (!generate)
							log_info_detail(info_detail);
#endif
						memory_errors++;
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
		if (votedOutput != NULL) 
			votedOutput[i] = valOutput;
		// if ((fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)||(fabs((tested_type_host)(valOutput-valGold)/valGold) > 1e-10)) {
		if (check) {
			if (valGold != valOutput) {
				if (checkFlag) {
#pragma omp critical
					{
						char error_detail[150];
						snprintf(error_detail, 150,
								"p: [%d, %d], r: %1.20e, e: %1.20e",
								(int) floor(i / k), i % k, (double)valOutput, (double)valGold);
						if (verbose && (host_errors < 10))
							printf("%s\n", error_detail);
#ifdef LOGS
						if (!generate)
							log_error_detail(error_detail);
#endif
						host_errors++;
					}
				}
			}
		}
	}

	// printf("numErrors:%d", host_errors);

#ifdef LOGS
	if (!generate) {
		log_info_count(memory_errors);
		log_error_count(host_errors);
	}
#endif
	if (memory_errors != 0) printf("M");
	if (host_errors != 0) printf("#");

	if ((host_errors != 0) || (host_is_memory_bad != 0)) { // (memory_errors != 0)
		//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
		freeCudaMemory();
		//====================================
		retrieveInputMatrices();
		//================== Init DEVICE memory
		allocCudaMemory();
		copyCudaMemory();
		//====================================
	}
	return (host_errors == 0) && (host_is_memory_bad == 0);
}

int main(int argc, char **argv)
{
//================== Test vars
	int loop2;
	// int kernel_errors=0;
	// int zero = 0;
	double time;
	double kernel_time, global_time;
	double total_kernel_time, min_kernel_time, max_kernel_time;
	int device_warmup = 1;
	
	// int gpu_check = 1;
	
//====================================

//================== Read test parameters
	if (argc < 2) {
		usage(argc, argv);
		exit(-1);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "size")) {
		k = getCmdLineArgumentInt(argc, (const char **) argv, "size");

		if ((k <= 0) || (k % 16 != 0)) {
			printf("Invalid input size given on the command-line: %d\n", k);
			exit (EXIT_FAILURE);
		}
		matrixSize = k * k;
	} else {
		usage(argc, argv);
		exit (EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_a",
				&a_matrix_path);
	} else {
		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "mxm_a_%s_%i.matrix",
				test_precision_description, (signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "mxm_b_%s_%i.matrix",
				test_precision_description, (signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", b_matrix_path);
	}
		
	if (checkCmdLineFlag(argc, (const char **) argv, "input_c")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_c",
				&c_matrix_path);
	} else {
		c_matrix_path = new char[100];
		snprintf(c_matrix_path, 100, "mxm_c_%s_%i.matrix",
				test_precision_description, (signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", c_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "mxm_gold_%s_%i.matrix", test_precision_description, (signed int) k);
		printf("Using default gold path: %s\n", gold_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		iterations = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		verbose = 1;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "debug")) {
		fault_injection = 1;
		printf("!! Will be injected an input error\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "no-warmup")) {
		device_warmup = 0;
		printf("!! The first iteration may not reflect real timing information\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		generate = 1;
		device_warmup = 0;
		fault_injection = 0;	//~ fault_injection = 1;
		iterations = 20;
		generate_safechecks = 5;
		printf("!! Generate !! Disabling device_warmup, fault_injection and iterations limiting.\n");
		printf("!! Generate parameters: generate_safechecks: %d / \n", generate_safechecks);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generator_debug")) {
		if (generate) {
			generator_debug = true;
		} else {
			printf("!! generator_debug ignored: generate is not activated. active with -generate.\n");
		}
	}
//====================================

//================== Set block and grid size for MxM kernel
#if defined(test_precision_double) or defined(test_precision_single)
	int gridsize = k / BLOCK_SIZE < 1 ? 1 : k / BLOCK_SIZE;
	int blocksize = k / BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(gridsize, gridsize);
#elif defined(test_precision_half)
	int gridsize = k / BLOCK_SIZE < 1 ? 1 : k / BLOCK_SIZE;
	int blocksize = k / BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize / 2.0, blocksize);
	dim3 dimGrid(gridsize, gridsize);
#endif
//====================================

//================== Init logs
#ifdef LOGS
	if (!generate) {
		char test_info[90];
		char test_name[90];
		snprintf(test_info, 90, "size:%d type:%s-precision-triplicated", k, test_precision_description);
		snprintf(test_name, 90, "cuda_trip_tensorcore_%s", test_precision_description);
		start_log_file(test_name, test_info);
	}
#endif
//====================================

//================== Alloc HOST memory
	A = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	B = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	C = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	D0 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	D1 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	D2 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));

	GOLD = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));

	if (!(A && B && C && D0 && D1 && D2 && GOLD)) {
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
	retrieveInputMatrices();
	printf("cuda_trip_tensorcore\n");
	fflush (stdout);
//====================================

//================== Init generator if enabled
	int generate_safechecks_count = 0;
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory();
//====================================

	for (loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

		host_is_memory_bad = 0;

		if (!loop2 && device_warmup)
			printf("First iteration: device warmup. Please wait...\n");

		global_time = mysecond();

		checkFrameworkErrors(
				cudaMemset(d_D0, 0, matrixSize * sizeof(tested_type)));
		checkFrameworkErrors(
				cudaMemset(d_D1, 0, matrixSize * sizeof(tested_type)));
		checkFrameworkErrors(
				cudaMemset(d_D2, 0, matrixSize * sizeof(tested_type)));

		checkFrameworkErrors( 
			cudaMemcpyToSymbol(is_memory_bad, &host_is_memory_bad,
				sizeof(unsigned long long int), 0, cudaMemcpyHostToDevice) );

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (!generate)
			if (loop2 || !device_warmup)
				start_iteration();
#endif
		//================== Device computation, MxM
		MatrixMulKernel<<<dimGrid,dimBlock>>>(	d_A0, d_A1, d_A2,
												d_B0, d_B1, d_B2, 
												d_C0, d_C1, d_C2,
												d_D0, d_D1, d_D2,
												k);

		checkFrameworkErrors(cudaPeekAtLastError());

		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError()); 
		//====================================
        
		
#ifdef LOGS
		if (!generate) 
			if (loop2 || !device_warmup)
				end_iteration();
#endif
		kernel_time = mysecond() - kernel_time;

		if (loop2 || !device_warmup) {
			total_kernel_time += kernel_time;
			min_kernel_time = min(min_kernel_time, kernel_time);
			max_kernel_time = max(max_kernel_time, kernel_time);
		}

		if (loop2 || !device_warmup)
			if (verbose)
				printf("Device kernel time for iteration %d: %.3fs\n", loop2,
						kernel_time);

		//================== Gold check
		if (verbose)
			printf(",");

		time = mysecond();

		if (loop2 || !device_warmup) {
			checkFrameworkErrors(
					cudaMemcpy(D0, d_D0, matrixSize * sizeof(tested_type),
							cudaMemcpyDeviceToHost));
			if ((generate) && (k <= 16)) {
				printf("\nMatrix D (0): \n");
				for (int i = 0; i<k*k; i++) {
					printf(" %.2e", (float)D0[i]);
					if ((i+1)%k == 0) printf("\n");
				}
				printf("\n");
			}

			checkFrameworkErrors(
					cudaMemcpy(D1, d_D1, matrixSize * sizeof(tested_type),
							cudaMemcpyDeviceToHost));
			if ((generate) && (k <= 16)) {
				printf("\nMatrix D (1): \n");
				for (int i = 0; i<k*k; i++) {
					printf(" %.2e", (float)D1[i]);
					if ((i+1)%k == 0) printf("\n");
				}
				printf("\n");
			}

			checkFrameworkErrors(
					cudaMemcpy(D2, d_D2, matrixSize * sizeof(tested_type),
							cudaMemcpyDeviceToHost));
			if ((generate) && (k <= 16)) {
				printf("\nMatrix D (2): \n");
				for (int i = 0; i<k*k; i++) {
					printf(" %.2e", (float)D2[i]);
					if ((i+1)%k == 0) printf("\n");
				}
				printf("\n");
			}

			checkFrameworkErrors(
				cudaMemcpyFromSymbol(&host_is_memory_bad, is_memory_bad,
						sizeof(unsigned long long int), 0,
						cudaMemcpyDeviceToHost) );
			if (verbose) {
				printf("is_memory_bad: %llu\n", host_is_memory_bad);
			}

			if (generate) {
				if (generate_safechecks_count == 0) {
					printf("Generate: First generation. Step %d/%d of max. %d \n", generate_safechecks_count, generate_safechecks, iterations);
					checkOutputErrors(GOLD, false); // This will copy the voted matrix to gold
					generate_safechecks_count++;
					if ((generate) && (k <= 16)) {
						printf("\nMatrix GOLD (VOTED): \n");
						for (int i = 0; i<k*k; i++) {
							printf(" %.2e", (float)GOLD[i]);
							if ((i+1)%k == 0) printf("\n");
						}
						printf("\n");
					}
				} else {
					if (!checkOutputErrors()) {
						printf("Generate: Failed on compare. Step %d/%d of max. %d \n", generate_safechecks_count, generate_safechecks, iterations);
						generate_safechecks_count = 0;
					} else {
						printf("Generate: Success on compare. Step %d/%d of max. %d\n", generate_safechecks_count, generate_safechecks, iterations);generate_safechecks_count++;
						if (generate_safechecks_count >= generate_safechecks) {
							writeGoldtoFile();
							loop2 = iterations; // This will make the loop end
						}
					}
				}
			} else {
				checkOutputErrors();
			}
		}
		//====================================

		//================== Console hearthbeat
		printf(".");
		fflush(stdout);
		//====================================

		if (loop2 || !device_warmup)
			if (verbose)
				printf("Gold check time for iteration %d: %.3fs\n", loop2,
						mysecond() - time);

		if (loop2 || !device_warmup)
			if (verbose) {
				/////////// PERF
				double flops = 2.0 * (double) k * k * k;
				double gflops = flops / kernel_time;
				double outputpersec = (double) matrixSize / kernel_time;
				printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", k,
						outputpersec, gflops, gflops / 1000000000);
				///////////
			}

		if (loop2 || !device_warmup)
			if (verbose)
				printf("Iteration #%d time: %.3fs\n\n\n", loop2,
						mysecond() - global_time);
		fflush(stdout);
	}

	double gflops = 2.0 * (double) k * k * k / 1000000000; // Bilion FLoating-point OPerationS
	double averageKernelTime = total_kernel_time
			/ (iterations - (device_warmup ? 1 : 0));
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

	free(A);
	free(B);
	free(C);
	free(D0);
	free(D1);
	free(D2);
	free(GOLD);
#ifdef LOGS
	if (!generate) 
		end_log_file();
#endif

//==================== ORIGINAL GEMM CODE ABOVE ===============
    
    //~ printf("Initializing...\n");

    //~ int dev = findCudaDevice(argc, (const char **)argv);

    //~ cudaDeviceProp deviceProp;
    //~ checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
    //~ if (deviceProp.major < 7) {
        //~ printf("cudaTensorCoreGemm requires requires SM 7.0 or higher to use Tensor Cores.  Exiting...\n");
        //~ exit(EXIT_WAIVED);
    //~ }

    //~ printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    //~ printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    //~ printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    //~ float *A_h = NULL;
    //~ float *B_h = NULL;
    //~ float *C_h = NULL;
//~ #if CPU_DEBUG
    //~ float *result_hD = NULL;
    //~ float *result_host = NULL;
//~ #endif

    //~ checkCudaErrors(cudaMallocManaged((void**)&A_h, sizeof(float) * M_GLOBAL * K_GLOBAL));
    //~ checkCudaErrors(cudaMallocManaged((void**)&B_h, sizeof(float) * K_GLOBAL * N_GLOBAL));
    //~ checkCudaErrors(cudaMallocManaged((void**)&C_h, sizeof(float) * M_GLOBAL * N_GLOBAL));
//~ #if CPU_DEBUG
    //~ checkCudaErrors(cudaMallocManaged((void**)&result_hD, sizeof(float) * M_GLOBAL * N_GLOBAL));
    //~ checkCudaErrors(cudaMallocManaged((void**)&result_host, sizeof(float) * M_GLOBAL * N_GLOBAL));
//~ #endif

    //~ half *A = NULL;
    //~ half *B = NULL;
    //~ float *C = NULL;
    //~ float *D = NULL;

    //~ checkCudaErrors(cudaMalloc((void**)&A, sizeof(half) * M_GLOBAL * K_GLOBAL));
    //~ checkCudaErrors(cudaMalloc((void**)&B, sizeof(half) * N_GLOBAL * K_GLOBAL));
    //~ checkCudaErrors(cudaMalloc((void**)&C, sizeof(float) * M_GLOBAL * N_GLOBAL));
    //~ checkCudaErrors(cudaMalloc((void**)&D, sizeof(float) * M_GLOBAL * N_GLOBAL));

    //~ assert(((unsigned long long)A) % 128 == 0);
    //~ assert(((unsigned long long)B) % 128 == 0);
    //~ assert(((unsigned long long)C) % 128 == 0);
    //~ assert(((unsigned long long)D) % 128 == 0);

    //init_host_matrices(A_h, B_h, C_h);

    //printf("Preparing data for GPU...\n");

    //checkKernelErrors((init_device_matrices<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK>>>(A_h, B_h, C_h, A, B, C, D)));

    //~ checkCudaErrors(cudaDeviceSynchronize());

    //~ enum {
        //~ // Compute the right amount of shared memory to request.
        //~ // We need shared memory to hold per-CTA C and D matrix tiles, and to cache per-CTA chunks
        //~ // of the A and B matrices. Therefore, the right amount to request is the maximum of those
        //~ // two numbers.
        //~ SHMEM_SZ = MAX(sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
                       //~ M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    //~ };

    //~ printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    //~ const float alpha = 1.1f;
    //~ const float beta = 1.2f;

    //~ cudaEvent_t start, stop;

    //~ checkCudaErrors(cudaEventCreate(&start));    
    //~ checkCudaErrors(cudaEventCreate(&stop));
    //~ checkCudaErrors(cudaEventRecord(start));

    // If enough shared memory available on the GPU use high performant kernel
    //~ if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)
    //~ {
        //~ printf("Computing... using high performance kernel compute_gemm \n");

       // checkCudaErrors(cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
       // checkKernelErrors((compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
       
       
		//~ MatrixMulKernel<<<gridDim,blockDim>>>(	d_A0, d_A1, d_A2,
												//~ d_B0, d_B1, d_B2, 
												//~ d_C0, d_C1, d_C2,
												//~ d_D0, d_D1, d_D2,
												//~ k);

		//~ checkFrameworkErrors(cudaPeekAtLastError());

		//~ checkFrameworkErrors(cudaDeviceSynchronize());
		//~ checkFrameworkErrors(cudaPeekAtLastError()); 
	//~ }	
       
       
//~ #if CPU_DEBUG
        //~ checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float)*M_GLOBAL*N_GLOBAL, cudaMemcpyDeviceToHost));
//~ #endif
    //~ }
    //~ else
    //~ {
        //~ dim3 gridDim;
        //~ dim3 blockDim;
     
        //~ // blockDim.x must be a multple of warpSize
        //~ // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        //~ blockDim.x = 128;
        //~ blockDim.y = 4;

        //~ gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        //~ gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        //printf("Computing... using simple_wmma_gemm kernel\n");
        //simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta);
          
		//~ MatrixMulKernel<<<gridDim,blockDim>>>(	d_A0, d_A1, d_A2,
												//~ d_B0, d_B1, d_B2, 
												//~ d_C0, d_C1, d_C2,
												//~ d_D0, d_D1, d_D2,
												//~ k);

		//~ checkFrameworkErrors(cudaPeekAtLastError());

		//~ checkFrameworkErrors(cudaDeviceSynchronize());
		//~ checkFrameworkErrors(cudaPeekAtLastError()); 
        
//~ #if CPU_DEBUG
        //~ checkCudaErrors(cudaMemcpy(result_hD, D, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
//~ #endif
    //~ }

    //~ checkCudaErrors(cudaEventRecord(stop));
    //~ checkCudaErrors(cudaEventSynchronize(stop));

//~ #if CPU_DEBUG
    //~ printf("Verifying correctness of the computations...\n");

    //~ memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

    //~ matMultiplyOnHost(A_h, B_h, result_host,
                      //~ alpha, beta,
                      //~ M_GLOBAL, K_GLOBAL,
                      //~ K_GLOBAL, N_GLOBAL,
                      //~ M_GLOBAL, N_GLOBAL);

    //~ for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
        //~ if (fabs(result_hD[i] - result_host[i]) > 0.1f)
            //~ printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i], result_host[i]);
    //~ }
//~ #endif

    //~ float milliseconds = 0;

    //~ checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    //~ printf("Time: %f ms\n", milliseconds);
    //~ printf("TFLOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2)/(milliseconds/1000.)) / 1e12);

    //~ checkCudaErrors(cudaFree((void*)A_h));
    //~ checkCudaErrors(cudaFree((void*)B_h));
    //~ checkCudaErrors(cudaFree((void*)C_h));
    //~ checkCudaErrors(cudaFree((void*)A));
    //~ checkCudaErrors(cudaFree((void*)B));
    //~ checkCudaErrors(cudaFree((void*)C));
    //~ checkCudaErrors(cudaFree((void*)D));

    return 0;
}
