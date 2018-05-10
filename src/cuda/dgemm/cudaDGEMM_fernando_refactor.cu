#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <string>

#ifdef USE_OMP
#include <omp.h>
#endif

#include <cublas.h>

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

#define DEFAULT_INPUT_SIZE 8192
#define ERROR_THRESHOLD  1.0e-10

#define GOLDCHK_BLOCK_SIZE 32

__device__ int kerrors;

typedef char byte;

////====================================

void get_device() {
//================== Retrieve and set the default CUDA device
	cudaDeviceProp prop;
	cudaError_t teste;
	int count = 0;
	teste = cudaGetDeviceCount(&count);
	printf("\nGet Device Test: %s\n", cudaGetErrorString(teste));
	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		printf("Name: %s\n", prop.name);
	}
	int *ndevice;
	int dev = 0;
	ndevice = &dev;
	cudaGetDevice(ndevice);

	cudaSetDevice(0);
	cudaGetDeviceProperties(&prop, 0);
	printf("\ndevice: %d %s\n", *ndevice, prop.name);

}

double mysecond() {
	struct timeval tp;
	struct timezone tzp;
	int i = gettimeofday(&tp, &tzp);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

void alloc_cuda_memory(double *d_a, double *d_b, double *d_c, int matrix_size) {
//================== CUDA error handlers
	cudaError_t malloc;
	const char *erro;
//====================================
	malloc = cudaMalloc((void**) &d_a, matrix_size * sizeof(double));
	erro = cudaGetErrorString(malloc);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error a"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc((void**) &d_b, matrix_size * sizeof(double));
	erro = cudaGetErrorString(malloc);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error b"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	malloc = cudaMalloc((void**) &d_c, matrix_size * sizeof(double));
	erro = cudaGetErrorString(malloc);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error c"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure
}

void copy_cuda_memory(double *d_a, double *d_b, double *d_c, double *a,
		double *b, int matrix_size) {
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================
	mcpy = cudaMemset(d_c, 0, matrix_size * sizeof(double));
	erro = cudaGetErrorString(mcpy);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load c"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	mcpy = cudaMemcpy(d_a, a, matrix_size * sizeof(double),
			cudaMemcpyHostToDevice); // PUSH A
	erro = cudaGetErrorString(mcpy);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load a"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	mcpy = cudaMemcpy(d_b, b, matrix_size * sizeof(double),
			cudaMemcpyHostToDevice); // PUSH B
	erro = cudaGetErrorString(mcpy);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		log_error_detail("error gpu load b"); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure
}

void read_matrix_from_file(FILE *f_A, FILE *f_B, FILE *f_GOLD, double *a,
		double *b, double *gold, char *a_matrix_path, char *b_matrix_path,
		char *gold_matrix_path, bool verbose, bool fault_injection, int k) {
//================== Read inputs to HOST memory
	if (verbose)
		printf("Reading matrices... ");
	double time = mysecond();
	f_A = fopen(a_matrix_path, "rb");
	f_B = fopen(b_matrix_path, "rb");
	f_GOLD = fopen(gold_matrix_path, "rb");
	if (!(f_A && f_B && f_GOLD)) {
		printf("Cant open matrices.\n");
#ifdef LOGS
		log_error_detail("Cant open matrices"); end_log_file();
#endif
		exit(-3);
	}
	size_t ret_value[3];
	for (int i = 0; i < k; i++) {
		ret_value[0] = fread(&(a[k * i]), sizeof(double) * k, 1, f_A);
		ret_value[1] = fread(&(b[k * i]), sizeof(double) * k, 1, f_B);
		ret_value[2] = fread(&(gold[k * i]), sizeof(double) * k, 1, f_GOLD);
		if ((ret_value[0] != 1) || (ret_value[1] != 1) || (ret_value[2] != 1)) {
			printf("Bad input/gold formatting: %lu ; %lu ; %lu .\n",
					ret_value[0], ret_value[1], ret_value[2]);
#ifdef LOGS
			log_error_detail("Bad input/gold formatting."); end_log_file();
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
		a[3] = (double) 6.5;
		printf("!! Injected 6.5 on position A[3]\n");
	}
}

__global__ void gold_check_kernel(double *gk, double *ck, int n) {
//================== HW Accelerated output validation
	int tx = blockIdx.x * GOLDCHK_BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * GOLDCHK_BLOCK_SIZE + threadIdx.y;
	//if ((fabs((gk[ty*n+tx]-ck[ty*n+tx])/gk[ty*n+tx]) > 0.0000000001)||(fabs((gk[ty*n+tx]-ck[ty*n+tx])/ck[ty*n+tx]) > 0.0000000001))
	if (gk[ty * n + tx] != ck[ty * n + tx])
		atomicAdd(&kerrors, 1);

}

void usage() {
	printf(
			"Usage: dgemm -size=N [-input_a=<path>] [-input_b=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup] [-gold_gpu_check]\n");
}

void check_output_errors(double *GOLD, double *d_A, double *d_B, double *d_C,
		double *C, int k, bool verbose) {
	char error_detail[150];
	int host_errors = 0;
#ifdef USE_OMP
#pragma omp parallel for
#endif
	for (int i = 0; (i < k); i++) {
		for (int j = 0; (j < k); j++) {
			double valGold = GOLD[i + k * j];
			double valOutput = C[i + k * j];
			// if ((fabs((double)(valOutput-valGold)/valGold) > 1e-10)||(fabs((double)(valOutput-valGold)/valGold) > 1e-10)) {
			if (valGold != valOutput) {
#ifdef USE_OMP
#pragma omp critical
#endif
				{
					snprintf(error_detail, 150,
							"p: [%d, %d], r: %1.20e, e: %1.20e", i, j,
							valOutput, valGold);
					if (verbose && (host_errors < 10))
						printf("%s\n", error_detail);
#ifdef LOGS
					log_error_detail(error_detail);
#endif
					host_errors++;
					//ea++;
					//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + k * j], GOLD[i + k * j], t_ea);
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
		cudaFree(d_A);
		cudaFree(d_B);
		cudaFree(d_C);
		//====================================
		read_matrix_from_file();
		//================== Init DEVICE memory
		alloc_cuda_memory();
		copy_cuda_memory();
		//====================================
	}
}

int main(int argc, char* argv[]) {
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================

	int verbose = 0;
	int fault_injection = 0;
	int gold_gpu_check = 0;

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
	double *C;
	double *GOLD;

	double *d_A;
	double *d_B;
	double *d_C;
	//====================================

//================== Test vars
	int loop2;
	// int kernel_errors=0;
	int zero = 0;
	double time;
	double kernel_time, global_time;
	double total_kernel_time, min_kernel_time, max_kernel_time;
	int device_warmup = 1;
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
			exit(EXIT_FAILURE);
		}
		matrixSize = k * k;
	} else {
		usage();
		exit(EXIT_FAILURE);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_a")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_a",
				&a_matrix_path);
	} else {
		a_matrix_path = new char[100];
		snprintf(a_matrix_path, 100, "dgemm_a_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", a_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "input_b")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input_b",
				&b_matrix_path);
	} else {
		b_matrix_path = new char[100];
		snprintf(b_matrix_path, 100, "dgemm_b_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input_a path: %s\n", b_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "dgemm_gold_%i.matrix", (signed int) k);
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

	if (checkCmdLineFlag(argc, (const char **) argv, "gold_gpu_check")) {
		gold_gpu_check = 1;
	}
//====================================

	////////////////////////////////////////////////////
	/////////////CUBLAS GEMM VARS///////////////////////
	const double alpha = 1.0;
	const double beta = 1.0;
	char transa = 't', transb = 't';
	////////////////////////////////////////////////////

//================== Init logs
#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d gpu-gold-check:%d type:double-precision", k, gold_gpu_check);
	start_log_file("cudaDGEMM", test_info);
#endif
//====================================

//================== Alloc HOST memory
	A = (double*) malloc(matrixSize * sizeof(double));
	B = (double*) malloc(matrixSize * sizeof(double));
	C = (double*) malloc(matrixSize * sizeof(double));

	GOLD = (double*) malloc(matrixSize * sizeof(double));

	if (!(A && B && C && GOLD)) {
		printf("Failed on host malloc.\n");
		exit(-3);
	}
//====================================

//================== Init test environment
	// kernel_errors=0;
	total_kernel_time = 0;
	min_kernel_time = UINT_MAX;
	max_kernel_time = 0;
	get_device();
	read_matrix_from_file();
	printf("cudaDGEMM\n");
	fflush(stdout);
//====================================

//================== Init DEVICE memory
	alloc_cuda_memory();
	copy_cuda_memory();
//====================================

	for (loop2 = 0; loop2 < iterations; loop2++) {//================== Global test loop

		if (!loop2 && device_warmup)
			printf("First iteration: device warmup. Please wait...\n");

		// Timer...
		global_time = mysecond();

		cudaMemset(d_C, 0, matrixSize * sizeof(double));
		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaPeekAtLastError());

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (loop2 || !device_warmup)
		start_iteration();
#endif
		//================== Device computation, HMxM

		cublasDgemm((cublasOperation_t) transa, (cublasOperation_t) transb, k,
				k, k, alpha, d_A, k, d_B, k, beta, d_C, k);

		checkCudaErrors(cudaPeekAtLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaPeekAtLastError());
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

		if (verbose)
			printf(",");

		// Timer...
		time = mysecond();

		//if (kernel_errors != 0) {
		if (loop2 || !device_warmup) {
			bool checkHost = true;
			if (gold_gpu_check == 1) {
				checkHost = false;

				//================== Set block and grid size for GoldChk kernel
				int gridsize =
						k / GOLDCHK_BLOCK_SIZE < 1 ? 1 : k / GOLDCHK_BLOCK_SIZE;
				int blocksize =
						k / GOLDCHK_BLOCK_SIZE < 1 ? k : GOLDCHK_BLOCK_SIZE;
				dim3 dimBlock(blocksize, blocksize);
				dim3 dimGrid(gridsize, gridsize);
				//====================================

				//================== Send GOLD to device, to perform HW output validation
				mcpy = cudaMemcpy(d_A, GOLD, matrixSize * sizeof(double),
						cudaMemcpyHostToDevice);
				erro = cudaGetErrorString(mcpy);
				if (strcmp(erro, "no error") != 0) {
					printf("error mem load gold\n");
#ifdef LOGS
					log_error_detail("error mem load gold"); end_log_file();
#endif
					return 1;
				} //mem allocate failure
				cudaMemcpyToSymbol(kerrors, &zero, sizeof(int));
				//====================================

				//================== Device computation, output validation
				gold_check_kernel<<<dimGrid, dimBlock>>>(d_A, d_C, k);
				checkCudaErrors(cudaPeekAtLastError());

				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaPeekAtLastError());
				//====================================

				//================== Retrieve output mismatchs
				unsigned int kernel_errors = 0;
				cudaMemcpyFromSymbol(&kernel_errors, kerrors,
						sizeof(unsigned int));
				if (kernel_errors != 0)
					checkHost = true;
				//====================================

				//================== Send A back to the device
				mcpy = cudaMemcpy(d_A, A, matrixSize * sizeof(double),
						cudaMemcpyHostToDevice);
				erro = cudaGetErrorString(mcpy);
				if (strcmp(erro, "no error") != 0) {
					printf("error mem load A\n");
#ifdef LOGS
					log_error_detail("error mem load A"); end_log_file();
#endif
					return 1;
				} //mem allocate failure
				  //====================================
			}
			if (checkHost) {
				checkCudaErrors(
						cudaMemcpy(C, d_C, matrixSize * sizeof(double),
								cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaPeekAtLastError());
				//~ if (memcmp(A, GOLD, sizeof(double) * k*k)) {
				check_output_errors(GOLD, d_A, d_B, d_C, C, k, verbose);
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
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	//====================================

	free(A);
	free(B);
	free(C);
	free(GOLD);
#ifdef LOGS
	end_log_file();
#endif

	return 0;
}
