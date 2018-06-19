#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <assert.h>
#include <random>
#include "cublas_v2.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef PRECISION_HALF
#include <cuda_fp16.h>
#include "half.hpp"
#endif

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

#ifdef SAFE_MALLOC
#include "safe_memory/safe_memory.h"
#else
#define safe_host_malloc malloc
#endif

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#undef min
#define min( x, y ) ( (x) < (y) ? (x) : (y) )
#undef max
#define max( x, y ) ( (x) > (y) ? (x) : (y) )

#ifndef DEFAULT_INPUT_SIZE
#define DEFAULT_INPUT_SIZE 8192
#endif

//=========== DEFINE TESTED TYPE
#if defined(PRECISION_DOUBLE)
	#define GENERATOR_MAXABSVALUE 4.1e+16
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "double";
	typedef double tested_type;
	typedef double tested_type_host;
#elif defined(PRECISION_SINGLE)
	#define GENERATOR_MAXABSVALUE 4.1e+2
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "single";
	typedef float tested_type;
	typedef float tested_type_host;
#elif defined(PRECISION_HALF)
	#define GENERATOR_MAXABSVALUE 2.0
	#define GENERATOR_MINABSVALUE 0
	const char test_precision_description[] = "half";
	typedef half tested_type;
	typedef half_float::half tested_type_host;
#else 
	#error TEST TYPE NOT DEFINED OR INCORRECT. USE PRECISION=<double|single|half>.
#endif

#ifndef checkFrameworkErrors
#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)
void __checkFrameworkErrors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail(errorDescription);
#endif

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

#define checkFrameworkErrorsNoFail(error) __checkFrameworkErrorsNoFail(error, __LINE__, __FILE__)
bool __checkFrameworkErrorsNoFail(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return false;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Not failing...",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail(errorDescription);
#endif

	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	return true;
}
#endif

//====================== benchmark+setup configuration
int generate = 0;
int verbose = 0;
int fault_injection = 0;
int test_input_check = 0;
int test_use_tensor = 0;
int test_gpu_check = 0;
#ifdef SAFE_MALLOC
int test_use_safemalloc = 1;
#else
int test_use_safemalloc = 0;
#endif

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
char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
tested_type_host *A, *endA;
tested_type_host *B, *endB;
tested_type_host *C;
tested_type_host *GOLD;

tested_type *d_A;
tested_type *d_B;
tested_type *d_C;
tested_type *d_GOLD;
//====================================

#define checkFrameworkErrors(error) __checkFrameworkErrors(error, __LINE__, __FILE__)
#define checkBlasFrameworkErrors(error) __checkBlasFrameworkErrors(error, __LINE__, __FILE__)

void __checkBlasFrameworkErrors(cublasStatus_t status, int line, const char* file) {
	if (status == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA cuBLAS Framework error: %d. Bailing.",
			status);
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

void allocCudaMemory() {

#ifdef SAFE_MALLOC
	d_A = (tested_type*) safe_malloc(matrixSize * sizeof(tested_type));
	d_B = (tested_type*) safe_malloc(matrixSize * sizeof(tested_type));
	d_C = (tested_type*) safe_malloc(matrixSize * sizeof(tested_type));
	if (test_gpu_check) {
		d_GOLD = (tested_type*) safe_malloc(matrixSize * sizeof(tested_type));
	}
#else
	checkFrameworkErrors(cudaMalloc(&d_A, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_B, matrixSize * sizeof(tested_type)));
	checkFrameworkErrors(cudaMalloc(&d_C, matrixSize * sizeof(tested_type)));
	if (test_gpu_check) {
		checkFrameworkErrors(cudaMalloc(&d_GOLD, matrixSize * sizeof(tested_type)));
	}
#endif

}

void freeCudaMemory() {
	checkFrameworkErrors(cudaFree(d_A));
	checkFrameworkErrors(cudaFree(d_B));
	checkFrameworkErrors(cudaFree(d_C));
	if (test_gpu_check) {
		checkFrameworkErrors(cudaFree(d_GOLD));
	}
}

void copyCudaMemory() {
	checkFrameworkErrors(cudaMemset(d_C, 0x00, matrixSize * sizeof(tested_type)));

	checkFrameworkErrors(
			cudaMemcpy(d_A, A, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH A

	checkFrameworkErrors(
			cudaMemcpy(d_B, B, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH B

	if (test_gpu_check) {
		checkFrameworkErrors(
			cudaMemcpy(d_GOLD, GOLD, matrixSize * sizeof(tested_type),
					cudaMemcpyHostToDevice)); // PUSH GOLD
	}
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
 		if (! (f_GOLD = fopen(gold_matrix_path, "rb"))) {
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
			ret_value[2] = fread(&(GOLD[k * i]), sizeof(tested_type) * k, 1, f_GOLD);
		}
		if ((ret_value[0] != 1) || (ret_value[1] != 1) || (gold && (ret_value[2] != 1))) {
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
	if (gold) fclose(f_GOLD);
}

void generateInputMatrices()
{
	FILE *f_A, *f_B;
	tested_type_host *h_A, *h_B;

	if (k==DEFAULT_INPUT_SIZE) {
		h_A = A;
		h_B = B;
	} else {
		h_A = (tested_type_host*) safe_host_malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		h_B = (tested_type_host*) safe_host_malloc(DEFAULT_INPUT_SIZE * DEFAULT_INPUT_SIZE * sizeof(tested_type));
		if (!(h_A && h_B)) {
			printf("Could not alloc h_A or h_B");
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
			}
		}
	} else {
		for (int i=0; i<DEFAULT_INPUT_SIZE; i++) {
			for (int j=0; j<DEFAULT_INPUT_SIZE; j++) {
				h_A[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)2.0;
				h_B[i * DEFAULT_INPUT_SIZE + j] = (tested_type_host)2.0;
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
	if (!(f_A&&f_B)) {
		printf("Could not open f_A or f_B\n");
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

	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_A[i * DEFAULT_INPUT_SIZE]), sizeof(tested_type) * DEFAULT_INPUT_SIZE, 1, f_A);
	}

	printf("Element 32 of matrix A: %f\n", (double)A[32]);

	printf("Element 50 of matrix B: %f\n", (double)B[50]);


	for(int i=0; i<DEFAULT_INPUT_SIZE; i++)
	{
		fwrite(&(h_B[i * DEFAULT_INPUT_SIZE]), sizeof(tested_type) * DEFAULT_INPUT_SIZE, 1, f_B);
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
		if (f_A) fclose(f_A);
		if (f_B) fclose(f_B);
		generateInputMatrices();
	} else {
		if (f_A) fclose(f_A);
		if (f_B) fclose(f_B);
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

	for(i=0; i<k; i++) {
		fwrite( &(GOLD[i * k]), sizeof(tested_type) * k, 1, f_GOLD );
	}

	fclose(f_GOLD);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
bool check_errors(bool check_output = true, bool check_input = true) {
	int output_errors = 0;
	int input_errors = 0;

#pragma omp parallel for shared(output_errors)
	for (int i = 0; i < matrixSize; i++) {
		if (check_output) {
			register tested_type_host valGold = GOLD[i];
			register tested_type_host valOutput = C[i];
			if (valGold != valOutput) {
#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150,
							"p: [%d, %d], r: %1.20e, e: %1.20e",
							(int) floor(i / k), i % k, (double)valOutput, (double)valGold);
					if (verbose && (output_errors < 10))
						printf("%s\n", error_detail);
#ifdef LOGS
					if (!generate)
						log_error_detail(error_detail);
#endif
					output_errors++;
				}
			}
		}
		if (check_input) {
			assert (endA != NULL);
			assert (endB != NULL);
			register tested_type_host valHostA = A[i];
			register tested_type_host valDeviceA = endA[i];
			register tested_type_host valHostB = B[i];
			register tested_type_host valDeviceB = endB[i];
			if (valHostA != valDeviceA) {
				#pragma omp critical
				{
					char info_detail[150];
					snprintf(info_detail, 150,
							"i: [%d, %d], m: A, r: %1.20e, e: %1.20e",
							(int) floor(i / k), i % k, (double)valDeviceA, (double)valHostA);
					if (verbose && (input_errors < 10))
						printf("%s\n", info_detail);
#ifdef LOGS
					if (!generate)
						log_info_detail(info_detail);
#endif
					input_errors++;
				}
			}
			if (valHostB != valDeviceB) {
				#pragma omp critical
				{
					char info_detail[150];
					snprintf(info_detail, 150,
							"i: [%d, %d], m: B, r: %1.20e, e: %1.20e",
							(int) floor(i / k), i % k, (double)valDeviceB, (double)valHostB);
					if (verbose && (input_errors < 10))
						printf("%s\n", info_detail);
#ifdef LOGS
					if (!generate)
						log_info_detail(info_detail);
#endif
					input_errors++;
				}
			}
		}
	}

	// printf("numErrors:%d", output_errors);

#ifdef LOGS
	if (!generate) {
		log_info_count(input_errors);
		log_error_count(output_errors);
	}
#endif
	if ((input_errors != 0) && (!verbose) ) printf("@");
	if ((input_errors != 0) && (verbose) ) printf("Input errors: %d\n", input_errors);
	if ((output_errors != 0) && (!verbose) ) printf("#");
	if ((output_errors != 0) && (verbose) ) printf("Output errors: %d\n", output_errors);

	if ((output_errors != 0) || (input_errors != 0)) {
		//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration
		freeCudaMemory();
		//====================================
		retrieveInputMatrices();
		//================== Init DEVICE memory
		allocCudaMemory();
		copyCudaMemory();
		//====================================
	}
	return (output_errors == 0) && (input_errors == 0);
}

///////////////////////////////////////// GOLD CHECK ON DEVICE ////////////////////////
#define GOLDCHK_BLOCK_SIZE 32
#define GOLDCHK_TILE_SIZE 16

__device__ unsigned long long int gck_device_errors;

__global__ void GoldChkKernel(tested_type *gk, tested_type *ck, int n) {
//================== HW Accelerated output validation
	int tx = (blockIdx.x * GOLDCHK_BLOCK_SIZE + threadIdx.x) * GOLDCHK_TILE_SIZE;
	int ty = (blockIdx.y * GOLDCHK_BLOCK_SIZE + threadIdx.y)  * GOLDCHK_TILE_SIZE;
	register unsigned int i, j, row;

#if defined(PRECISION_DOUBLE) or defined(PRECISION_SINGLE)
	for (i=ty; i<ty+GOLDCHK_TILE_SIZE; i++) {
		row = i * n;
		for (j=tx; j<tx+GOLDCHK_TILE_SIZE; j++) {
			if (gk[row + j] != ck[row + j]) {
				atomicAdd(&gck_device_errors, 1);
			}
		}
	}
#elif defined(PRECISION_HALF)
	for (i=ty; i<ty+GOLDCHK_TILE_SIZE; i++) {
		row = i * n;
		for (j=tx; j<tx+GOLDCHK_TILE_SIZE; j+=2) {
			if (__hbne2(*((half2*)(&(gk[row + j]))), *((half2*)(&(ck[row + j]))))) {
				atomicAdd(&gck_device_errors, 1);
			}
		}
	}
#endif

}
///////////////////////////////////////// GOLD CHECK ON DEVICE ////////////////////////

void usage(int argc, char* argv[]) {
	printf("Usage: %s -size=N [-generate] [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-gpu_check] [-test_input_check] [-no-warmup] [-use_tensor=<0|1>] [-input_check=<0|1>]\n", argv[0]);
}

int main(int argc, char* argv[]) {
//================== Test vars
	int loop2;
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
			exit (EXIT_FAILURE);
		}
		if (k>DEFAULT_INPUT_SIZE) {
			printf("\n========>> Warning:\nSIZE > DEFAULT_INPUT_SIZE. May crash on input retrieval.\n============\n");
		}
		matrixSize = k * k;
	} else {
		usage(argc, argv);
		exit (EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "use_tensor")) {
		test_use_tensor = getCmdLineArgumentInt(argc, (const char **) argv, "use_tensor");
#ifdef PRECISION_DOUBLE
		if (test_use_tensor) {
			test_use_tensor = false;
			printf("\n========>> Warning:\nNo tensor cores are available in Double precision\n============\n");
		}
#endif
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_a",
				&a_matrix_path);
	} else {
		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "gemm_%s_A_%i_%s.matrix",
			test_precision_description, (signed int) DEFAULT_INPUT_SIZE, test_use_tensor ? "tensor" : "normal");
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "gemm_%s_B_%i_%s.matrix",
			test_precision_description, (signed int) DEFAULT_INPUT_SIZE, test_use_tensor ? "tensor" : "normal");
		printf("Using default input_a path: %s\n", b_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "gemm_%s_GOLD_%i_%s.matrix", 
			test_precision_description, (signed int) k, test_use_tensor ? "tensor" : "normal");
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
	
	if (checkCmdLineFlag(argc, (const char **) argv, "gpu_check")) {
		if (k % (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE) == 0) {
			test_gpu_check = true;
			printf("> Testing on GPU using GoldChk kernel\n");
		} else {
			test_gpu_check = false;
			printf("\n========>> Warning:\nCannot test on GPU with this size. Size must be a multiple of %d.\n============\n", (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE));
		}
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		generate = 1;
		device_warmup = 0;
		fault_injection = 0;
		iterations = 20;
		generate_safechecks = 5;
		test_input_check = true;
		test_gpu_check = false;
		printf("!! Generate !! Enabling input_check. Disabling device_warmup, gpu check, fault_injection and iterations limiting.\n");
		printf("!! Generate parameters: generate_safechecks: %d / \n", generate_safechecks);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generator_debug")) {
		if (generate) {
			generator_debug = true;
		} else {
			printf("!! generator_debug ignored: generate is not activated. active with -generate.\n");
		}
	}
	
	if (checkCmdLineFlag(argc, (const char **) argv, "input_check")) {
		test_input_check = getCmdLineArgumentInt(argc, (const char **) argv, "input_check");
	}
//====================================

//================== Set cuBLAS GEMM parameters
////////////////////////////////////////////////////////////////////
	const tested_type_host alpha = tested_type_host(1.0);
	const tested_type_host beta = tested_type_host(1.0);
	cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_T;
////////////////////////////////////////////////////////////////////
//==================

//================== Init logs
	char test_info[150];
	char test_name[50];
	snprintf(test_info, 150, "size:%d precision:%s tensor_cores:%d safe_malloc:%d gpu_check:%d test_input_check:%d", 
		k, test_precision_description, test_use_tensor, test_use_safemalloc, test_gpu_check, test_input_check);
	snprintf(test_name, 50, "cuda_%s_gemm", test_precision_description);
	printf("%s\n%s\n", test_name, test_info);
#ifdef LOGS
	if (!generate) {
		start_log_file(test_name, test_info);
	}
#endif
//====================================

//================== Alloc HOST memory
	A = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));
	B = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));
	C = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));

	GOLD = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));

	if (!(A && B && C && GOLD)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}

	if (test_input_check) {
		endA = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));
		endB = (tested_type_host*) safe_host_malloc(matrixSize * sizeof(tested_type));
		if (!(endA && endB)) {
			printf("Failed on host malloc.\n");
			exit(-3);
		}
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
	total_kernel_time = 0;
	min_kernel_time = UINT_MAX;
	max_kernel_time = 0;
	GetDevice();
	retrieveInputMatrices();
	printf("cuda_gemm\n");
	fflush (stdout);

////////////////////////////////////////////////////////////////////
// cuBLAS Creation and setup
	cublasHandle_t blas_handle;

	checkBlasFrameworkErrors( cublasCreate(&blas_handle) );

	if (test_use_tensor) {
		checkBlasFrameworkErrors( cublasSetMathMode(blas_handle, CUBLAS_TENSOR_OP_MATH) );
	} else {
		checkBlasFrameworkErrors( cublasSetMathMode(blas_handle, CUBLAS_DEFAULT_MATH) );
	}
////////////////////////////////////////////////////////////////////

////////////// GOLD CHECK Kernel /////////////////
	dim3 gck_blockSize = dim3(	GOLDCHK_BLOCK_SIZE, 
								GOLDCHK_BLOCK_SIZE);
	dim3 gck_gridSize = dim3(	k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE), 
								k / (GOLDCHK_BLOCK_SIZE * GOLDCHK_TILE_SIZE));
//////////////////////////////////////////////////
//====================================

//================== Init generator if enabled
	int generate_safechecks_count = 0;
	bool generate_success = false;
//====================================

//================== Init DEVICE memory
	allocCudaMemory();
	copyCudaMemory();
//====================================

	for (loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

		if (!loop2 && device_warmup)
			printf("First iteration: device warmup. Please wait...\n");

		global_time = mysecond();

		checkFrameworkErrors(
				cudaMemset(d_C, 0, matrixSize * sizeof(tested_type)));

		checkFrameworkErrors( cudaDeviceSynchronize() );

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (!generate)
			if (loop2 || !device_warmup)
				start_iteration();
#endif
		//================== Device computation, gemm
#if defined(PRECISION_DOUBLE)
		checkBlasFrameworkErrors( cublasDgemm(blas_handle,
			transa, 
			transb, 
			k, k, k,
			(tested_type*)&alpha, 
			d_A, k, 
			d_B, k, 
			(tested_type*)&beta, 	
			d_C, k) );
#elif defined(PRECISION_SINGLE)
		checkBlasFrameworkErrors( cublasSgemm(blas_handle,
			transa, 
			transb, 
			k, k, k,
			(tested_type*)&alpha, 
			d_A, k, 
			d_B, k, 
			(tested_type*)&beta, 	
			d_C, k) );
#elif defined(PRECISION_HALF)
		checkBlasFrameworkErrors( cublasHgemm(blas_handle,
			transa, 
			transb, 
			k, k, k,
			(tested_type*)&alpha, 
			d_A, k, 
			d_B, k, 
			(tested_type*)&beta, 	
			d_C, k) );
#endif

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
			bool checkOnHost = false;
			if (test_gpu_check) {
				assert (d_GOLD != NULL);

				// Send to device
				unsigned long long int gck_errors = 0;
				checkOnHost |= checkFrameworkErrorsNoFail( cudaMemcpyToSymbol(gck_device_errors, &gck_errors, sizeof(unsigned long long int)) );
				// GOLD is already on device.

				/////////////////// Run kernel
				GoldChkKernel<<<gck_gridSize, gck_blockSize>>>(d_GOLD, d_C, k);
				checkOnHost |= checkFrameworkErrorsNoFail( cudaPeekAtLastError() );
				checkOnHost |= checkFrameworkErrorsNoFail( cudaDeviceSynchronize() );
				///////////////////

				// Receive from device
				checkOnHost |= checkFrameworkErrorsNoFail( cudaMemcpyFromSymbol(&gck_errors, gck_device_errors, sizeof(unsigned long long int)) );
				if (gck_errors != 0) {
					printf("$(%u)", (unsigned int)gck_errors);
					checkOnHost = true;
				}
			} else {
				checkOnHost = true;
			}
			if (checkOnHost) {
				checkFrameworkErrors(
					cudaMemcpy(C, d_C, matrixSize * sizeof(tested_type),
							cudaMemcpyDeviceToHost));
				if ((generate) && (k <= 16)) {
					printf("\nMatrix C: \n");
					for (int i = 0; i<k*k; i++) {
						printf(" %.2e", (float)C[i]);
						if ((i+1)%k == 0) printf("\n");
					}
					printf("\n");
				}

				if (test_input_check) {
					checkFrameworkErrors(
						cudaMemcpy(endA, d_A, matrixSize * sizeof(tested_type),
								cudaMemcpyDeviceToHost));
					if ((generate) && (k <= 16)) {
						printf("\nMatrix A(end): \n");
						for (int i = 0; i<k*k; i++) {
							printf(" %.2e", (float)endA[i]);
							if ((i+1)%k == 0) printf("\n");
						}
						printf("\n");
					}

					checkFrameworkErrors(
							cudaMemcpy(endB, d_B, matrixSize * sizeof(tested_type),
									cudaMemcpyDeviceToHost));
					if ((generate) && (k <= 16)) {
						printf("\nMatrix B(end): \n");
						for (int i = 0; i<k*k; i++) {
							printf(" %.2e", (float)endB[i]);
							if ((i+1)%k == 0) printf("\n");
						}
						printf("\n");
					}
				}


				if (generate) {
					if (generate_safechecks_count == 0) {
						if (verbose) printf("Generate: First generation. Step %d/%d of max. %d \n", generate_safechecks_count, 
						generate_safechecks, iterations);
						if (check_errors(false, test_input_check) ) {
							generate_safechecks_count++;
							memcpy(GOLD, C, matrixSize * sizeof(tested_type));
						}
					} else {
						if (!check_errors(true, test_input_check)) {
							if (verbose) printf("Generate: Failed on compare. Step %d/%d of max. %d \n", generate_safechecks_count, generate_safechecks, iterations);
							generate_safechecks_count = 0;
						} else {
							if (verbose) printf("Generate: Success on compare. Step %d/%d of max. %d\n", generate_safechecks_count, generate_safechecks, iterations);generate_safechecks_count++;
							if (generate_safechecks_count >= generate_safechecks) {
								generate_success = true;
								writeGoldtoFile();
								break;
							}
						}
					}
				} else {
					check_errors(true, test_input_check);
				}
			}
		}
		//====================================

		//================== Console hearthbeat
		if (!verbose) printf(".");
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
			/ (loop2 - (device_warmup && !generate ? 1 : 0));
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
	free(GOLD);
	if (endA) free(endA);
	if (endB) free(endB);
#ifdef LOGS
	if (!generate) 
		end_log_file();
#endif

	if (generate && !generate_success) {
		exit(EXIT_FAILURE);
	}

	return 0;
}
