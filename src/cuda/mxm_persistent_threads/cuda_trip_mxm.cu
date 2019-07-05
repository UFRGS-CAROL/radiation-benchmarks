#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <omp.h>
#include <random>
#include <cuda_fp16.h>

#include <memory>

#ifdef LOGS
#include "log_helper.h"

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif

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
tested_type *d_C0; //, *d_C1, *d_C2;

tested_type *d_A0; //, *d_A1, *d_A2;
tested_type *d_B0; //, *d_B1, *d_B2;
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

void allocCudaMemory() {
	rad::checkFrameworkErrors(
			cudaMalloc(&d_A0, matrixSize * sizeof(tested_type)));
	rad::checkFrameworkErrors(
			cudaMalloc(&d_B0, matrixSize * sizeof(tested_type)));
#ifndef CONSTINPUT
	rad::checkFrameworkErrors(
			cudaMalloc(&d_C0, matrixSize * sizeof(tested_type)));
#endif

}

void freeCudaMemory() {
	rad::checkFrameworkErrors(cudaFree(d_A0));
	rad::checkFrameworkErrors(cudaFree(d_B0));

	rad::checkFrameworkErrors(cudaFree(d_C0));
}

void copyCudaMemory() {
	//clear C
	rad::checkFrameworkErrors(
			cudaMemset(d_C0, 0x00, matrixSize * sizeof(tested_type)));

	rad::checkFrameworkErrors(
			cudaMemcpy(d_A0, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH A

	rad::checkFrameworkErrors(
			cudaMemcpy(d_B0, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH B
}

void readMatricesFromFile(bool gold = true) {
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
	ret_value[0] = fread(A, sizeof(tested_type), matrixSize, f_A);
	ret_value[1] = fread(B, sizeof(tested_type), matrixSize, f_B);
	if (gold) {
		ret_value[2] = fread(GOLD, sizeof(tested_type), matrixSize, f_GOLD);
	}
	if ((ret_value[0] != matrixSize) || (ret_value[1] != matrixSize)
			|| (gold && (ret_value[2] != matrixSize))) {
		printf("Bad input/gold formatting: %lu ; %lu ; %lu .\n", ret_value[0],
				ret_value[1], ret_value[2]);
#ifdef LOGS
		if (!generate)
		log_error_detail((char *)"Bad input/gold formatting."); end_log_file();
#endif
		exit(-3);
	}

	fclose(f_A);
	fclose(f_B);
	if (gold)
		fclose(f_GOLD);
}

void generateInputMatrices() {
	FILE * f_A, *f_B;
	tested_type *h_A = A;
	tested_type *h_B = B;

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(-GENERATOR_MAXABSVALUE,
			GENERATOR_MAXABSVALUE);

	if (!generator_debug) {
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				h_A[i * k + j] = (tested_type_host) dis(gen);
				h_B[i * k + j] = (tested_type_host) dis(gen);
			}
		}
	} else {
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < k; j++) {
				h_A[i * k + j] = (tested_type_host) 2.0;
				h_B[i * k + j] = (tested_type_host) 2.0;
			}
		}
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
	for (int i = 0; i < matrixSize; i++) {
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
	for (int i = 0; i < matrixSize; i++) {
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
//
//	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
//		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]),
//				sizeof(tested_type) * DEFAULT_INPUT_SIZE, 1, f_A);
//	}
	fwrite(h_A, sizeof(tested_type), matrixSize, f_A);
	printf("Element 32 of matrix A: %f\n", (double) A[32]);

	printf("Element 50 of matrix B: %f\n", (double) B[50]);

//	for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
//		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]),
//				sizeof(tested_type_host) * DEFAULT_INPUT_SIZE, 1, f_B);
//	}
	fwrite(h_B, sizeof(tested_type), matrixSize, f_B);

	printf("Done\n");

	fclose(f_A);
	fclose(f_B);
	return;
}

void retrieveInputMatrices() {
//================== Read inputs to HOST memory
	double time = rad::mysecond();

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
		printf("Done reading matrices in %.2fs\n", rad::mysecond() - time);
}

void writeGoldtoFile() {
	f_GOLD = fopen(gold_matrix_path, "wb");
	if (!f_GOLD) {
		printf("Could not open f_GOLD\n");
		exit(EXIT_FAILURE);
	}

	fwrite(GOLD, sizeof(tested_type), k * k, f_GOLD);

	fclose(f_GOLD);
}

template<typename real>
__device__ void process_data(int wA, int wB, const real* A, const real* B,
		real* C) {
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
template<typename real> __global__ void MatrixMulKernel(real *A, real *B,
		real *C, int wA, int wB) {
	rad::PersistentKernel pk;
	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			process_data(wA, wB, A, B, C);
			pk.iteration_finished();
		}
	}
}

void usage(int argc, char* argv[]) {
	printf(
			"Usage: %s -size=N [-generate] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n",
			argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool checkOutputErrors() {
	int host_errors = 0;
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < matrixSize; i++) {
		register tested_type_host valGold = GOLD[i];
		register tested_type_host valOutput = C0[i];

		if (valGold != valOutput) {
#pragma omp critical
			{
				char error_detail[150];
				snprintf(error_detail, 150, "p: [%d, %d], r: %1.20e, e: %1.20e",
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

#ifdef LOGS
	if (!generate) {
		log_error_count(host_errors);
	}
#endif

	if (host_errors != 0)
		printf("#");

	return (host_errors == 0);
}

void launch_kernel(dim3 dimGrid, dim3 dimBlock, cudaStream_t stream) {
//Starting persistent kernel
	MatrixMulKernel<<<dimGrid, dimBlock, 0, stream>>>(d_A0, d_B0, d_C0, k, k);
	rad::checkFrameworkErrors (cudaPeekAtLastError());printf
	("Kernel LAUCHED\n");
}

int main(int argc, char* argv[]) {
//================== Test vars
	double time;
	double kernel_time, global_time;
	double total_kernel_time, min_kernel_time, max_kernel_time;
	int device_warmup = 1;
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
				test_precision_description, (signed int) k);
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "mxm_b_%s_%i.matrix",
				test_precision_description, (signed int) k);
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
		char test_info[150];
		char test_name[150];
		snprintf(test_info, 150, "size:%d type:%s-precision block_size:%d", k,
				test_precision_description, BLOCK_SIZE);
		snprintf(test_name, 150, "cuda_%s_mxm_persistent_threads", test_precision_description);
		start_log_file(test_name, test_info);
		set_iter_interval_print(30);
	}

	std::string log_file_name(get_log_file_name());
	if(generate) {
		log_file_name = "/tmp/generate.log";
	}
//	rad::Profiler profiler_thread = new rad::JTX2Inst(log_file_name);
	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);

//START PROFILER THREAD
	profiler_thread->start_profile();
#endif
//====================================

//================== Alloc HOST memory
	A = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	B = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	GOLD = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	if (!(A && B)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
// kernel_errors=0;
	retrieveInputMatrices();

	C0 = (tested_type_host*) malloc(matrixSize * sizeof(tested_type));
	if (!(C0 && GOLD)) { //&& C1 && C2
		printf("Failed on host malloc.\n");
		exit(-3);
	}

	total_kernel_time = 0;
	min_kernel_time = UINT_MAX;
	max_kernel_time = 0;
	GetDevice();
	printf("cuda_%s_mxm\n", test_precision_description);
	fflush(stdout);
//====================================

//================== Init generator if enabled
//====================================
//================== Init Persistent threads controler
	rad::HostPersistentControler pt_control(dimGrid);

//====================================
//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory();
//====================================
//Starting persistent kernel
	cudaStream_t stream;
	rad::checkFrameworkErrors(
			cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	launch_kernel(dimGrid, dimBlock, stream);

	for (size_t loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

		global_time = rad::mysecond();
		rad::checkFrameworkErrors(
				cudaMemset(d_C0, 0, sizeof(tested_type) * matrixSize));
		if (verbose)
			printf(",");

		kernel_time = rad::mysecond();
#ifdef LOGS
		if (!generate)
		start_iteration();
#endif
		//================== Device computation, MxM

		pt_control.process_data_on_kernel();

		rad::checkFrameworkErrors (cudaPeekAtLastError());
		//====================================
#ifdef LOGS
		if (!generate)
		end_iteration();
#endif
kernel_time		= rad::mysecond() - kernel_time;

		total_kernel_time += kernel_time;
		min_kernel_time = min(min_kernel_time, kernel_time);
		max_kernel_time = max(max_kernel_time, kernel_time);

		if (verbose) {
			printf("Device kernel time for iteration %ld: %.3fs\n", loop2,
					kernel_time);

			//================== Gold check
			printf(",");
		}
		time = rad::mysecond();
		rad::checkFrameworkErrors(
				cudaMemcpy(C0, d_C0, matrixSize * sizeof(tested_type),
						cudaMemcpyDeviceToHost));

		if ((generate) && (k <= 16)) {
			printf("\nMatrix C (0): \n");
			for (int i = 0; i < k * k; i++) {
				printf(" %.2e", (float) C0[i]);
				if ((i + 1) % k == 0)
					printf("\n");
			}
			printf("\n");
		}

		if (generate) {
			memcpy(GOLD, C0, sizeof(tested_type) * matrixSize);
			writeGoldtoFile();
			break;
		} else {
			bool executed_ok = checkOutputErrors();
			if (executed_ok == false) { // (memory_errors != 0)
#ifdef LOGS
					profiler_thread->end_profile();
#endif
				pt_control.end_kernel();

				//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
				freeCudaMemory();
				//====================================
				retrieveInputMatrices();
				//================== Init DEVICE memory
				allocCudaMemory();
				copyCudaMemory();

				//====================================
#ifdef LOGS
				profiler_thread->start_profile();
#endif
				// Re-launch the kernel
				pt_control.start_kernel();
				launch_kernel(dimGrid, dimBlock, stream);
			}
		}
//		}
		//====================================

		//================== Console hearthbeat
		printf(".");
		fflush(stdout);
		//====================================
		if (verbose) {
			printf("Gold check time for iteration %ld: %.3fs\n", loop2,
					rad::mysecond() - time);
			/////////// PERF
			double flops = 2.0 * (double) k * k * k;
			double gflops = flops / kernel_time;
			double outputpersec = (double) matrixSize / kernel_time;
			printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", k,
					outputpersec, gflops, gflops / 1000000000);
			///////////

			printf("Iteration #%ld time: %.3fs\n\n\n", loop2,
					rad::mysecond() - global_time);
		}
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
	rad::checkFrameworkErrors(cudaStreamDestroy(stream));
//================== Release device memory
	freeCudaMemory();

	free(A);
	free(B);
	free(GOLD);
	free(C0);

#ifdef LOGS
	if (!generate)
	end_log_file();
#endif

	return 0;
}
