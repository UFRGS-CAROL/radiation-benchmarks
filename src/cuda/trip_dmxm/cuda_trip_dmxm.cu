#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>
#include <omp.h>

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

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

int verbose = 0;
int fault_injection = 0;

int k = 0; // k x k matrix size
int matrixSize = 0; // = k * k matrix size
int iterations = 100000000; // global loop iteracion

//================== Input paths
char *gold_matrix_path, *a_matrix_path, *b_matrix_path;

FILE* f_A;
FILE* f_B;
FILE* f_GOLD;
//====================================

//================== Host and device matrix ptr's
double *A;
double *B;
double *C[3];
double *GOLD;

double *d_A[3];
double *d_B[3];
double *d_C[3];
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
	d_A[0] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_A[1] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_A[2] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));

	d_B[0] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_B[1] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_B[2] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));

	d_C[0] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_C[1] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
	d_C[2] = (double*) safe_cudaMalloc(matrixSize * sizeof(double));
}

void freeCudaMemory() {
	checkFrameworkErrors(cudaFree(d_A[0]));
	checkFrameworkErrors(cudaFree(d_A[1]));
	checkFrameworkErrors(cudaFree(d_A[2]));

	checkFrameworkErrors(cudaFree(d_B[0]));
	checkFrameworkErrors(cudaFree(d_B[1]));
	checkFrameworkErrors(cudaFree(d_B[2]));

	checkFrameworkErrors(cudaFree(d_C[0]));
	checkFrameworkErrors(cudaFree(d_C[1]));
	checkFrameworkErrors(cudaFree(d_C[2]));
}

void copyCudaMemory() {
	checkFrameworkErrors(cudaMemset(d_C[0], 0x00, matrixSize * sizeof(double)));
	checkFrameworkErrors(cudaMemset(d_C[1], 0x00, matrixSize * sizeof(double)));
	checkFrameworkErrors(cudaMemset(d_C[2], 0x00, matrixSize * sizeof(double)));

	checkFrameworkErrors(
			cudaMemcpy(d_A[0], A, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH A
	checkFrameworkErrors(
			cudaMemcpy(d_A[1], A, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH A
	checkFrameworkErrors(
			cudaMemcpy(d_A[2], A, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH A

	checkFrameworkErrors(
			cudaMemcpy(d_B[0], B, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH B
	checkFrameworkErrors(
			cudaMemcpy(d_B[1], B, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH B
	checkFrameworkErrors(
			cudaMemcpy(d_B[2], B, matrixSize * sizeof(double),
					cudaMemcpyHostToDevice)); // PUSH B
}

void ReadMatrixFromFile() {
//================== Read inputs to HOST memory
	int i;
	if (verbose)
		printf("Reading matrices... ");
	double time = mysecond();
	f_A = fopen(a_matrix_path, "rb");
	f_B = fopen(b_matrix_path, "rb");
	f_GOLD = fopen(gold_matrix_path, "rb");
	if (!(f_A && f_B && f_GOLD)) {
		printf("Cant open matrices.\n");
#ifdef LOGS
		log_error_detail((char *)"Cant open matrices"); end_log_file();
#endif
		exit(-3);
	}
	size_t ret_value[3];
	for (i = 0; i < k; i++) {
		ret_value[0] = fread(&(A[k * i]), sizeof(double) * k, 1, f_A);
		ret_value[1] = fread(&(B[k * i]), sizeof(double) * k, 1, f_B);
		ret_value[2] = fread(&(GOLD[k * i]), sizeof(double) * k, 1, f_GOLD);
		if ((ret_value[0] != 1) || (ret_value[1] != 1) || (ret_value[2] != 1)) {
			printf("Bad input/gold formatting: %lu ; %lu ; %lu .\n",
					ret_value[0], ret_value[1], ret_value[2]);
#ifdef LOGS
			log_error_detail((char *)"Bad input/gold formatting."); end_log_file();
#endif
			exit(-3);
		}
	}
	if (verbose)
		printf("Done reading matrices in %.2fs\n", mysecond() - time);

	fclose(f_A);
	fclose(f_B);
	fclose(f_GOLD);

	if (fault_injection) {
		A[3] = (double) 6.5;
		printf("!! Injected 6.5 on position A[3]\n");
	}
}

__global__ void MatrixMulKernel(double *d_A0, double *d_A1, double *d_A2,
		double *d_B0, double *d_B1, double *d_B2, double *d_C0, double *d_C1,
		double *d_C2, int n) {
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int k;

	register size_t offset_A;
	register double in_A;
	register double in_A1;

	register size_t offset_B;
	register double in_B;
	register double in_B1;

	register double acc = 0.0;
	for (k = 0; k < n; k++) {

		offset_A = ty * n + k;
		in_A = d_A0[offset_A];
		in_A1 = d_A1[offset_A];
		if (in_A != in_A1) {
			if (in_A != d_A2[offset_A]) {
				in_A = in_A1;
			}
		}

		offset_B = k * n + tx;
		in_B = d_B0[offset_B];
		in_B1 = d_B1[offset_B];
		if (in_B != in_B1) {
			if (in_B != d_B2[offset_B]) {
				in_B = in_B1;
			}
		}

		acc += in_A * in_B;
	}

	register size_t offset_C = ty * n + tx;

	d_C0[offset_C] = acc;
	d_C1[offset_C] = acc;
	d_C2[offset_C] = acc;
}

void usage() {
	printf(
			"Usage: cuda_trip_dmxm -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

void checkOutputErrors() {
	int host_errors = 0;
	int memory_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < matrixSize; i++) {
		register bool checkFlag = true;
		register double valGold = GOLD[i];
		register double valOutput0 = C[0][i];
		register double valOutput1 = C[1][i];
		register double valOutput2 = C[2][i];
		register double valOutput = valOutput0;
		if ((valOutput0 != valOutput1) || (valOutput1 != valOutput2)) {
#pragma omp critical
			{
				char info_detail[150];
				snprintf(info_detail, 150,
						"m: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e",
						(int) floor(i / k), i % k, valOutput0, valOutput1,
						valOutput2);
				if (verbose && (memory_errors < 10))
					printf("%s\n", info_detail);

#ifdef LOGS
				log_info_detail(info_detail);
#endif
				memory_errors += 1;
			}
			if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2)) {
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
						char error_detail[150];
						snprintf(error_detail, 150,
								"f: [%d, %d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e",
								(int) floor(i / k), i % k, valOutput0,
								valOutput1, valOutput2, valGold);
						if (verbose && (host_errors < 10))
							printf("%s\n", error_detail);

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
		if (valGold != valOutput) {
			if (checkFlag) {
#pragma omp critical
				{
					char error_detail[150];
					snprintf(error_detail, 150,
							"p: [%d, %d], r: %1.20e, e: %1.20e",
							(int) floor(i / k), i % k, valOutput, valGold);
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
		//====================================
		ReadMatrixFromFile();
		//================== Init DEVICE memory
		allocCudaMemory();
		copyCudaMemory();
		//====================================
	}
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
		usage();
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
		usage();
		exit (EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_a",
				&a_matrix_path);
	} else {
		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "dmxm_a_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "dmxm_b_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", b_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "dmxm_gold_%i.matrix", (signed int) k);
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
//====================================

//================== Set block and grid size for MxM kernel
	int gridsize = k / BLOCK_SIZE < 1 ? 1 : k / BLOCK_SIZE;
	int blocksize = k / BLOCK_SIZE < 1 ? k : BLOCK_SIZE;
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(gridsize, gridsize);
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d type:double-precision-triplicated", k);
	start_log_file((char *)"cuda_trip_dmxm", test_info);
#endif
//====================================

//================== Alloc HOST memory
	A = (double*) malloc(matrixSize * sizeof(double));
	B = (double*) malloc(matrixSize * sizeof(double));
	C[0] = (double*) malloc(matrixSize * sizeof(double));
	C[1] = (double*) malloc(matrixSize * sizeof(double));
	C[2] = (double*) malloc(matrixSize * sizeof(double));

	GOLD = (double*) malloc(matrixSize * sizeof(double));

	if (!(A && B && C[0] && C[1] && C[2] && GOLD)) {
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
	printf("cuda_trip_dmxm\n");
	fflush (stdout);
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
				cudaMemset(d_C[0], 0, matrixSize * sizeof(double)));
		checkFrameworkErrors(
				cudaMemset(d_C[1], 0, matrixSize * sizeof(double)));
		checkFrameworkErrors(
				cudaMemset(d_C[2], 0, matrixSize * sizeof(double)));

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (loop2 || !device_warmup)
		start_iteration();
#endif
		//================== Device computation, DMxM
		MatrixMulKernel<<<dimGrid, dimBlock>>>(d_A[0], d_A[1], d_A[2], d_B[0],
				d_B[1], d_B[2], d_C[0], d_C[1], d_C[2], k);

		checkFrameworkErrors(cudaPeekAtLastError());

		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());
		//====================================
#ifdef LOGS
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
					cudaMemcpy(C[0], d_C[0], matrixSize * sizeof(double),
							cudaMemcpyDeviceToHost));
			checkFrameworkErrors(
					cudaMemcpy(C[1], d_C[1], matrixSize * sizeof(double),
							cudaMemcpyDeviceToHost));
			checkFrameworkErrors(
					cudaMemcpy(C[2], d_C[2], matrixSize * sizeof(double),
							cudaMemcpyDeviceToHost));
			checkOutputErrors();
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
	free(C[0]);
	free(C[1]);
	free(C[2]);
	free(GOLD);
#ifdef LOGS
	end_log_file();
#endif

	return 0;
}
