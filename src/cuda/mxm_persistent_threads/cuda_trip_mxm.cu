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

#include "include/persistent_lib.h"
// The timestamp is updated on every log_helper function call.

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#include "half.hpp"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

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

//unsigned long long int host_is_memory_bad = 0;

int k = 0; // k x k matrix size
int matrixSize = 0; // = k * k matrix size
int iterations = 100000000; // global loop iteration
//=========================

//======== generator configuration
int generate_safechecks = 0;
bool generate_inputmatricesready = false;
bool host_check = false;
bool generator_debug = false;
//=========================

//================== Input paths
char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
tested_type_host *A;
tested_type_host *B;
tested_type_host *C0; //, *C1, *C2;
tested_type_host *GOLD;

tested_type *d_A0; //, *d_A1, *d_A2;
tested_type *d_B0; //, *d_B1, *d_B2;
tested_type *d_C0; //, *d_C1, *d_C2;
//====================================

void GetDevice() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	int count = 0;
	printf("Get device:");
	rad::checkFrameworkErrors(cudaGetDeviceCount(&count));
	for (int i = 0; i < count; i++) {
		rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
		printf("Name: %s\n", prop.name);
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	rad::checkFrameworkErrors(cudaGetDevice(ndevice));

	rad::checkFrameworkErrors(cudaSetDevice(0));
	rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));
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
	rad::checkFrameworkErrors(cudaMalloc(&devicePtr, size));
	outputPtr = malloc(size);
	goldPtr = malloc(size);
	if ((outputPtr == NULL) || (goldPtr == NULL)) {
		log_error_detail((char *) "error host malloc");
		end_log_file();
		printf("error host malloc\n");
		exit(EXIT_FAILURE);
	}

	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	rad::checkFrameworkErrors(cudaMemset(devicePtr, 0xAA, size));
	memset(goldPtr, 0xAA, size);

	rad::checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		rad::checkFrameworkErrors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END FIRST PHASE

	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	rad::checkFrameworkErrors(cudaMemset(devicePtr, 0x55, size));
	memset(goldPtr, 0x55, size);

	rad::checkFrameworkErrors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		rad::checkFrameworkErrors(cudaFree(devicePtr));
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
//	d_A1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
//	d_A2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));

	d_B0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
//	d_B1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
//	d_B2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));

	d_C0 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
//	d_C1 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
//	d_C2 = (tested_type*) safe_cudaMalloc(matrixSize * sizeof(tested_type));
#else
	rad::checkFrameworkErrors(cudaMalloc(&d_A0, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_A1, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_A2, matrixSize * sizeof(tested_type)));

	rad::checkFrameworkErrors(cudaMalloc(&d_B0, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_B1, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_B2, matrixSize * sizeof(tested_type)));

	rad::checkFrameworkErrors(cudaMalloc(&d_C0, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_C1, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(cudaMalloc(&d_C2, matrixSize * sizeof(tested_type)));
#endif

}

void freeCudaMemory() {
	rad::checkFrameworkErrors(cudaFree(d_A0));
//	rad::checkFrameworkErrors(cudaFree(d_A1));
//	rad::checkFrameworkErrors(cudaFree(d_A2));

	rad::checkFrameworkErrors(cudaFree(d_B0));
//	rad::checkFrameworkErrors(cudaFree(d_B1));
//	rad::checkFrameworkErrors(cudaFree(d_B2));

	rad::checkFrameworkErrors(cudaFree(d_C0));
//	rad::checkFrameworkErrors(cudaFree(d_C1));
//	rad::checkFrameworkErrors(cudaFree(d_C2));
}

void copyCudaMemory(rad::HostPersistentControler& pt_control) {
	rad::checkFrameworkErrors(
			cudaMemsetAsync(d_C0, 0x00, matrixSize * sizeof(tested_type),
					pt_control.st));
//	rad::checkFrameworkErrors(
//			cudaMemset(d_C1, 0x00, matrixSize * sizeof(tested_type)));
//	rad::checkFrameworkErrors(
//			cudaMemset(d_C2, 0x00, matrixSize * sizeof(tested_type)));
	pt_control.sync_stream();

	rad::checkFrameworkErrors(
			cudaMemcpyAsync(d_A0, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice, pt_control.st)); // PUSH A
//	rad::checkFrameworkErrors(
//			cudaMemcpy(d_A1, A, matrixSize * sizeof(tested_type),
//					cudaMemcpyHostToDevice)); // PUSH A
//	rad::checkFrameworkErrors(
//			cudaMemcpy(d_A2, A, matrixSize * sizeof(tested_type),
//					cudaMemcpyHostToDevice)); // PUSH A
	pt_control.sync_stream();

	rad::checkFrameworkErrors(
			cudaMemcpyAsync(d_B0, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice, pt_control.st)); // PUSH B
//	rad::checkFrameworkErrors(
//			cudaMemcpy(d_B1, B, matrixSize * sizeof(tested_type),
//					cudaMemcpyHostToDevice)); // PUSH B
//	rad::checkFrameworkErrors(
//			cudaMemcpy(d_B2, B, matrixSize * sizeof(tested_type),
//					cudaMemcpyHostToDevice)); // PUSH B
	pt_control.sync_stream();
}

void readMatricesFromFile(bool gold = true) {
	int i;
	f_A = fopen(a_matrix_path, "rb");
	f_B = fopen(b_matrix_path, "rb");
	if (!(f_A && f_B)) {
		printf("Cant open input  matrices.\n");
#ifdef LOGS
		if (!generate)
		log_error_detail((char *)"Cant open input matrices"); end_log_file();
#endif
		exit(-3);
	}
	if (gold) {
		if (!(f_GOLD = fopen(gold_matrix_path, "rb"))) {
			printf("Cant open gold matrice.\n");
#ifdef LOGS
			if (!generate)
			log_error_detail((char *)"Cant open gold matrice"); end_log_file();
#endif
			exit(-3);
		}
	}

	size_t ret_value[3];
	for (i = 0; i < k; i++) {
		ret_value[0] = fread(&(A[k * i]), sizeof(tested_type) * k, 1, f_A);
		ret_value[1] = fread(&(B[k * i]), sizeof(tested_type) * k, 1, f_B);
		if (gold) {
			ret_value[2] = fread(&(GOLD[k * i]), sizeof(tested_type) * k, 1,
					f_GOLD);
		}
		if ((ret_value[0] != 1) || (ret_value[1] != 1)
				|| (gold && (ret_value[2] != 1))) {
			printf("Bad input/gold formatting: %lu ; %lu ; %lu .\n",
					ret_value[0], ret_value[1], ret_value[2]);
#ifdef LOGS
			if (!generate)
			log_error_detail((char *)"Bad input/gold formatting."); end_log_file();
#endif
			exit(-3);
		}
	}

	fclose(f_A);
	fclose(f_B);
	if (gold)
		fclose(f_GOLD);
}

void generateInputMatrices() {
	FILE * f_A, *f_B;
	tested_type_host *h_A, *h_B;

	if (k == DEFAULT_INPUT_SIZE) {
		h_A = A;
		h_B = B;
	} else {
		h_A = (tested_type_host*) malloc(
		DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		h_B = (tested_type_host*) malloc(
		DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		if (!(h_A && h_B)) {
			printf("Could not alloc h_A or h_B");
			exit(EXIT_FAILURE);
		}
	}

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(-GENERATOR_MAXABSVALUE,
			GENERATOR_MAXABSVALUE);

	if (!generator_debug) {
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
			for (int j = 0; j < DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host) dis(gen);
				h_B[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host) dis(gen);
			}
		}
	} else {
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
			for (int j = 0; j < DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host) 2.0;
				h_B[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host) 2.0;
			}
		}
	}

	if (h_A != A) {
		memcpy(A, h_A, matrixSize * sizeof(tested_type));
		memcpy(B, h_B, matrixSize * sizeof(tested_type));
	}

	int numZeros;
	int numNans;
	int numInfs;
// printf("Write\n");
	f_A = fopen(a_matrix_path, "wb");
	f_B = fopen(b_matrix_path, "wb");
	if (!(f_A && f_B)) {
		printf("Could not open f_A or f_B\n");
		exit(EXIT_FAILURE);
	}

	tested_type_host val;

	numZeros = 0;
	numNans = 0;
	numInfs = 0;
	for (int i = 0; i < DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE; i++) {
		val = h_A[i];
		if (val == 0)
			numZeros++;
		if (isnan(val))
			numNans++;
		if (isinf(val))
			numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix A: %d/%d/%d\n", numZeros,
			numNans, numInfs);

	numZeros = 0;
	numNans = 0;
	numInfs = 0;
	for (int i = 0; i < DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE; i++) {
		val = h_B[i];
		if (val == 0)
			numZeros++;
		if (isnan(val))
			numNans++;
		if (isinf(val))
			numInfs++;
	}
	printf("Number of zeros/NaNs/INFs on matrix B: %d/%d/%d\n", numZeros,
			numNans, numInfs);

	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]),
				sizeof(tested_type) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f\n", (double) A[32]);

	printf("Element 50 of matrix B: %f\n", (double) B[50]);

	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]),
				sizeof(tested_type_host) * DEFAULT_INPUT_SIZE, 1, f_B);
	}
	printf("Done\n");

	fclose(f_A);
	fclose(f_B);
	if (h_A != A) {
		free(h_A);
		free(h_B);
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
	if (generate && (!f_A || !f_B)) {
		if (f_A)
			fclose(f_A);
		if (f_B)
			fclose(f_B);
		generateInputMatrices();
	} else {
		if (f_A)
			fclose(f_A);
		if (f_B)
			fclose(f_B);
		readMatricesFromFile(!generate);
	}

	if ((generate) && (generator_debug) && (k <= 16)) {
		printf("\nMatrix A: \n");
		for (int i = 0; i < k * k; i++) {
			printf(" %.2e", (float) A[i]);
			if ((i + 1) % k == 0)
				printf("\n");
		}
		printf("\nMatrix B: \n");
		for (int i = 0; i < k * k; i++) {
			printf(" %.2e", (float) B[i]);
			if ((i + 1) % k == 0)
				printf("\n");
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

	for (i = 0; i < k; i++) {
		fwrite(&(GOLD[i * k]), sizeof(tested_type) * k, 1, f_GOLD);
	}

	fclose(f_GOLD);
}

template<typename real>
__device__ void process_data(int wA, int wB, real* A, real* B, real* C) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;
	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;
	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;
	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;
	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;
	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	real Csub = 0;
	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ real As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ real Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}
	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template<typename real> __global__ void MatrixMulKernel(real *C, real *A,
		real *B, int wA, int wB) {
	rad::PersistentKernel pk;
	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			process_data(wA, wB, A, B, C);
			pk.iteration_finished();
		}
	}
}

//__global__ void MatrixMulKernel(half *d_A0, half *d_B0, half *d_C0, int wA,
//		int wB) {
//	register int tx = (blockIdx.x * BLOCK_SIZE) / 2.0 + threadIdx.x;
//	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
//	register int k;
//
//	register half2 acc = __float2half2_rn(0.0);
//	for (k = 0; k < wA; k++) {
//
//		acc = __hfma2(__half2half2(d_A0[ty * wA + k]),
//				__half2half2(d_B0[k * (wA / 2) + tx]), acc);
//		// n/2 is needed because we changed how we iterate d_B
//	}
//
//	((half2*) d_C0)[ty * (wA / 2) + tx] = acc;
//}

void usage(int argc, char* argv[]) {
	printf(
			"Usage: %s -size=N [-generate] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n",
			argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool checkOutputErrors(tested_type_host* votedOutput = NULL,
		bool check = true) {
	int host_errors = 0;
//	int memory_errors = 0;

//	if (host_is_memory_bad != 0) {
//		char info_detail[150];
//		snprintf(info_detail, 150, "b: is_memory_bad: %llu",
//				host_is_memory_bad);
//		if (verbose)
//			printf("%s\n", info_detail);
//
//#ifdef LOGS
//		if (!generate)
//		log_info_detail(info_detail);
//#endif
//		memory_errors++;
//	}

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < matrixSize; i++) {
		register bool checkFlag = true;
		register tested_type_host valGold = GOLD[i];
		register tested_type_host valOutput = C0[i];

//		register tested_type_host valOutput0 = C0[i];
//		register tested_type_host valOutput1 = C0[i];
//		register tested_type_host valOutput2 = C0[i];
//		if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
//#pragma omp critical
//			{
//				char info_detail[150];
//				snprintf(info_detail, 150,
//						"m: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e",
//						(int) floor(i / k), i % k, (double) valOutput0,
//						(double) valOutput1, (double) valOutput2);
//				if (verbose && (memory_errors < 10))
//					printf("%s\n", info_detail);
//
//#ifdef LOGS
//				if (!generate)
//				log_info_detail(info_detail);
//#endif
//				memory_errors++;
//			}
//			if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2)
//					&& (valOutput0 != valOutput2)) {
//				// All 3 values diverge
//				if (valOutput0 == valGold) {
//					valOutput = valOutput0;
//				} else if (valOutput1 == valGold) {
//					valOutput = valOutput1;
//				} else if (valOutput2 == valGold) {
//					valOutput = valOutput2;
//				} else {
//					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
//					checkFlag = false;
//#pragma omp critical
//					{
//						char info_detail[150];
//						snprintf(info_detail, 150,
//								"t: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e",
//								(int) floor(i / k), i % k, (double) valOutput0,
//								(double) valOutput1, (double) valOutput2,
//								(double) valGold);
//						if (verbose && (memory_errors < 10))
//							printf("%s\n", info_detail);
//
//#ifdef LOGS
//						if (!generate)
//						log_info_detail(info_detail);
//#endif
//						memory_errors++;
//					}
//				}
//			} else if (valOutput1 == valOutput2) {
//				// Only value 0 diverge
//				valOutput = valOutput1;
//			} else if (valOutput0 == valOutput2) {
//				// Only value 1 diverge
//				valOutput = valOutput0;
//			} else if (valOutput0 == valOutput1) {
//				// Only value 2 diverge
//				valOutput = valOutput0;
//			}
//		}

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
								(int) floor(i / k), i % k, (double) valOutput,
								(double) valGold);
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
//		log_info_count(memory_errors);
		log_error_count(host_errors);
	}
#endif

	if (host_errors != 0)
		printf("#");

	return (host_errors == 0);
}

void launch_kernel(dim3 dimGrid, dim3 dimBlock) {
	//Starting persistent kernel
	MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A0, d_B0, d_C0, k, k);
	rad::checkFrameworkErrors(cudaPeekAtLastError());
	printf("Kernel LAUCHED\n");
}

int main(int argc, char* argv[]) {
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
			exit(EXIT_FAILURE);
		}
		matrixSize = k * k;
	} else {
		usage(argc, argv);
		exit(EXIT_FAILURE);
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

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "mxm_gold_%s_%i.matrix",
				test_precision_description, (signed int) k);
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
		printf(
				"!! The first iteration may not reflect real timing information\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		generate = 1;
		device_warmup = 0;
		fault_injection = 0;
		iterations = 20;
		generate_safechecks = 5;
		printf(
				"!! Generate !! Disabling device_warmup, fault_injection and iterations limiting.\n");
		printf("!! Generate parameters: generate_safechecks: %d / \n",
				generate_safechecks);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generator_debug")) {
		if (generate) {
			generator_debug = true;
		} else {
			printf(
					"!! generator_debug ignored: generate is not activated. active with -generate.\n");
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
		snprintf(test_info, 90, "size:%d type:%s-precision", k, test_precision_description);
		snprintf(test_name, 90, "cuda_%s_mxm", test_precision_description);
		start_log_file(test_name, test_info);
	}
#endif
//====================================

//================== Alloc HOST memory
	A = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	B = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	C0 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
//	C1 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
//	C2 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));

	GOLD = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));

	if (!(A && B && C0 && GOLD)) { //&& C1 && C2
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
	printf("cuda_%s_mxm\n", test_precision_description);
	fflush(stdout);
//====================================

//================== Init generator if enabled
	int generate_safechecks_count = 0;
//====================================
//================== Init Persistent threads controler
	rad::HostPersistentControler pt_control(dimGrid);

//====================================
//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory(pt_control);
//====================================
	//Starting persistent kernel
	launch_kernel(dimGrid, dimBlock);

	for (loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

//		host_is_memory_bad = 0;

		if (!loop2 && device_warmup)
			printf("First iteration: device warmup. Please wait...\n");

		global_time = mysecond();

		rad::checkFrameworkErrors(
				cudaMemsetAsync(d_C0, 0, matrixSize * sizeof(tested_type),
						pt_control.st));
		pt_control.sync_stream();
		printf("CUDAMEMSET ISSUED\n");
//		rad::checkFrameworkErrors(
//				cudaMemset(d_C1, 0, matrixSize * sizeof(tested_type)));
//		rad::checkFrameworkErrors(
//				cudaMemset(d_C2, 0, matrixSize * sizeof(tested_type)));

//		rad::checkFrameworkErrors(
//				cudaMemcpyToSymbol(is_memory_bad, &host_is_memory_bad,
//						sizeof(unsigned long long int), 0,
//						cudaMemcpyHostToDevice));

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (!generate)
		if (loop2 || !device_warmup)
		start_iteration();
#endif
		//================== Device computation, MxM

		pt_control.process_data_on_kernel();

//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
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
			rad::checkFrameworkErrors(
					cudaMemcpyAsync(C0, d_C0, matrixSize * sizeof(tested_type),
							cudaMemcpyDeviceToHost, pt_control.st));
			pt_control.sync_stream();
			if ((generate) && (k <= 16)) {
				printf("\nMatrix C (0): \n");
				for (int i = 0; i < k * k; i++) {
					printf(" %.2e", (float) C0[i]);
					if ((i + 1) % k == 0)
						printf("\n");
				}
				printf("\n");
			}

//			rad::checkFrameworkErrors(
//					cudaMemcpy(C1, d_C1, matrixSize * sizeof(tested_type),
//							cudaMemcpyDeviceToHost));
//			if ((generate) && (k <= 16)) {
//				printf("\nMatrix C (1): \n");
//				for (int i = 0; i < k * k; i++) {
//					printf(" %.2e", (float) C1[i]);
//					if ((i + 1) % k == 0)
//						printf("\n");
//				}
//				printf("\n");
//			}
//
//			rad::checkFrameworkErrors(
//					cudaMemcpy(C2, d_C2, matrixSize * sizeof(tested_type),
//							cudaMemcpyDeviceToHost));
//			if ((generate) && (k <= 16)) {
//				printf("\nMatrix C (2): \n");
//				for (int i = 0; i < k * k; i++) {
//					printf(" %.2e", (float) C2[i]);
//					if ((i + 1) % k == 0)
//						printf("\n");
//				}
//				printf("\n");
//			}
//			rad::checkFrameworkErrors(
//					cudaMemcpyFromSymbol(&host_is_memory_bad, is_memory_bad,
//							sizeof(unsigned long long int), 0,
//							cudaMemcpyDeviceToHost));
//			if (verbose) {
//				printf("is_memory_bad: %llu\n", host_is_memory_bad);
//			}

			if (generate) {
				if (generate_safechecks_count == 0) {
					printf(
							"Generate: First generation. Step %d/%d of max. %d \n",
							generate_safechecks_count, generate_safechecks,
							iterations);
					checkOutputErrors(GOLD, false); // This will copy the voted matrix to gold
					generate_safechecks_count++;
					if ((generate) && (k <= 16)) {
						printf("\nMatrix GOLD (VOTED): \n");
						for (int i = 0; i < k * k; i++) {
							printf(" %.2e", (float) GOLD[i]);
							if ((i + 1) % k == 0)
								printf("\n");
						}
						printf("\n");
					}
				} else {
					if (!checkOutputErrors()) {
						printf(
								"Generate: Failed on compare. Step %d/%d of max. %d \n",
								generate_safechecks_count, generate_safechecks,
								iterations);
						generate_safechecks_count = 0;
					} else {
						printf(
								"Generate: Success on compare. Step %d/%d of max. %d\n",
								generate_safechecks_count, generate_safechecks,
								iterations);
						generate_safechecks_count++;
						if (generate_safechecks_count >= generate_safechecks) {
							writeGoldtoFile();
							loop2 = iterations; // This will make the loop end
							//stops only after generating
							pt_control.end_kernel();

						}
					}
				}
			} else {
				bool executed_ok = checkOutputErrors();
				if (executed_ok == false) { // (memory_errors != 0)
					pt_control.end_kernel();

					//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
					freeCudaMemory();
					//====================================
					retrieveInputMatrices();
					//================== Init DEVICE memory
					allocCudaMemory();
					copyCudaMemory(pt_control);
					//====================================
					// Re-launch the kernel
					pt_control.start_kernel();
					launch_kernel(dimGrid, dimBlock);
				}
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

	pt_control.end_kernel();

	//================== Release device memory
	freeCudaMemory();
	//====================================

	free(A);
	free(B);
	free(C0);
//	free (C1);
//	free (C2);
	free(GOLD);
#ifdef LOGS
	if (!generate)
	end_log_file();
#endif

	return 0;
}
