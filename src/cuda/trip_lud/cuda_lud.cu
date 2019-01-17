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
#include <math.h>
#include <string>

// helper functions
#include "helper_string.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "lud_kernel.h"

#ifdef LOGS
#include "log_helper.h"
#endif

#define DEFAULT_INPUT_SIZE 8192

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

template<typename real_t, typename real_t_device>
void copy_cuda_memory(real_t_device *d_OUTPUT, real_t_device *d_INPUT,
		real_t *INPUT, int matrixSize, int generate) {
//================== CUDA error handlers
	cudaError_t mcpy;
	const char *erro;
//====================================
	mcpy = cudaMemset(d_OUTPUT, 0, matrixSize * sizeof(real_t_device));
	erro = cudaGetErrorString(mcpy);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail(const_cast<char*>("error gpu output load memset")); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	mcpy = cudaMemcpy(d_INPUT, INPUT, matrixSize * sizeof(real_t_device),
			cudaMemcpyHostToDevice); // PUSH A
	erro = cudaGetErrorString(mcpy);
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail(const_cast<char*>("error gpu load input")); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure
}

template<typename real_t>
void generate_input_matrix(real_t *m, char *input_matrix_path) {
	FILE *f_INPUT;
	if (!(f_INPUT = fopen(input_matrix_path, "wb"))) {
		printf("Error: Could not open input file in wb mode. %s\n",
				input_matrix_path);
		exit(EXIT_FAILURE);
	} else {
		printf("Generating input matrix of size %dx%d...\n", DEFAULT_INPUT_SIZE,
		DEFAULT_INPUT_SIZE);
		real_t tempArray[DEFAULT_INPUT_SIZE];
		for (int i = 0; i < DEFAULT_INPUT_SIZE; i++) {
#pragma omp parallel for
			for (int j = 0; j < DEFAULT_INPUT_SIZE; j++)
				tempArray[j] = (real_t) rand() / 32768.0;
			size_t ret_value = 0;
			ret_value = fwrite(tempArray, DEFAULT_INPUT_SIZE * sizeof(real_t),
					1, f_INPUT);
			if (ret_value != 1) {
				printf("Failure writing to input: %ld\n", ret_value);
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_INPUT);
	}
}

template<typename real_t>
void write_gold_file(real_t *m, char *gold_matrix_path, int k) {
	FILE *f_GOLD;
	if (!(f_GOLD = fopen(gold_matrix_path, "wb"))) {
		printf("Error: Could not open gold file in wb mode. %s\n",
				gold_matrix_path);
		exit(EXIT_FAILURE);
	} else {
		size_t ret_value = 0;
		for (int i = 0; i < k; i++) {
			ret_value = fwrite(&(m[i * k]), k * sizeof(real_t), 1, f_GOLD);
			if (ret_value != 1) {
				printf("Failure writing to gold: %ld\n", ret_value);
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_GOLD);
	}
}

template<typename real_t>
void read_matrix_from_file(real_t *INPUT, real_t *GOLD, char *input_matrix_path,
		char *gold_matrix_path, int verbose, int generate, int k,
		int fault_injection) {
//================== Read inputs to HOST memory
	int i;

	if (verbose)
		printf("Reading matrices...");
	double time = mysecond();

	if (generate) {
		generate_input_matrix<real_t>(INPUT, input_matrix_path);
	}
	FILE *f_INPUT = fopen(input_matrix_path, "rb");

	if (f_INPUT) {
		// open input successful
		size_t ret_value;
		for (i = 0; i < k; i++) {
			ret_value = fread(&(INPUT[k * i]), sizeof(real_t) * k, 1, f_INPUT);
			if (ret_value != 1) {
				printf("Bad input formatting: %lu .\n", ret_value);
#ifdef LOGS
				log_error_detail(const_cast<char*>("Bad input formatting.")); end_log_file();
#endif
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_INPUT);
	} else {
		printf("Cant open matrices and -generate is false.\n");
#ifdef LOGS
		log_error_detail(const_cast<char*>("Cant open matrices")); end_log_file();
#endif
		exit(EXIT_FAILURE);
	}

	FILE *f_GOLD;
	if (!generate) {
		size_t ret_value;
		f_GOLD = fopen(gold_matrix_path, "rb");
		for (i = 0; i < k; i++) {
			ret_value = fread(&(GOLD[k * i]), sizeof(real_t) * k, 1, f_GOLD);
			if (ret_value != 1) {
				printf("Bad gold formatting: %lu .\n", ret_value);
#ifdef LOGS
				log_error_detail(const_cast<char*>("Bad gold formatting.")); end_log_file();
#endif
				exit(EXIT_FAILURE);
			}
		}
		fclose(f_GOLD);
	}
	if (verbose)
		printf("Done reading matrices in %.2fs\n", mysecond() - time);

	if (fault_injection) {
		INPUT[3] = (float) 6.5;
		printf("!! Injected 6.5 on position INPUT[3]\n");
	}
}

#pragma omp declare reduction(+: half_float::half : omp_out += omp_in)

template<typename real_t>
bool badass_memcmp(real_t *gold, real_t *found, unsigned long n) {
	real_t result = real_t(0.0);
	int i;
	unsigned long chunk = ceil(real_t(n) / real_t(omp_get_max_threads()));
	// printf("size %d max threads %d chunk %d\n", n, omp_get_max_threads(), chunk);
	double time = mysecond();
#pragma omp parallel for default(shared) private(i) schedule(static,chunk) reduction(+:result)
	for (i = 0; i < n; i++)
		result = result + (gold[i] - found[i]);

	//  printf("comparing took %lf seconds, diff %lf\n", mysecond() - time, result);
	if (fabs(result) > 0.0000000001)
		return true;
	return false;
}

void usage() {
	printf(
			"Usage: lud -size=N [-generate] [-input=<path>] [-gold=<path>] [-iterations=N] [-verbose] [-no-warmup]\n");
}

template<typename real_t, typename real_t_device>
void test_lud_radiation(int matrixSize, int verbose, int generate, int k,
		int fault_injection, int iterations, int device_warmup,
		char* input_matrix_path, char* gold_matrix_path,
		std::string precision) {
	//====================================
	double time;
	double kernel_time, global_time;
	double total_kernel_time, min_kernel_time, max_kernel_time;

	//================== Alloc HOST memory
	real_t* INPUT = (real_t*) (malloc(matrixSize * sizeof(real_t)));
	real_t* OUTPUT = (real_t*) (malloc(matrixSize * sizeof(real_t)));
	real_t* GOLD = (real_t*) (malloc(matrixSize * sizeof(real_t)));

	if (!(INPUT && GOLD && OUTPUT)) {
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
	read_matrix_from_file<real_t>(INPUT, GOLD, input_matrix_path,
			gold_matrix_path, verbose, generate, k, fault_injection);
	fflush(stdout);
	//====================================

#ifdef LOGS
	char test_info[90];
	snprintf(test_info, 90, "size:%d type:%s-precision", k, precision.c_str());
	std::string benchmark = std::string("cuda") + precision.c_str() + "LUD";
	if (!generate) start_log_file(const_cast<char*>(benchmark.c_str()), test_info);
#endif

	//================== Init DEVICE memory
	real_t_device* d_INPUT;
	real_t_device* d_OUTPUT;
	const char *erro = cudaGetErrorString(
			cudaMalloc((void**) &d_INPUT, matrixSize * sizeof(real_t_device)));
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail(const_cast<char*>("error input")); end_log_file();
#endif
		exit(EXIT_FAILURE);
	}
	erro = cudaGetErrorString(
			cudaMalloc((void**) &d_OUTPUT, matrixSize * sizeof(real_t_device)));
	if (strcmp(erro, "no error") != 0) {
#ifdef LOGS
		if (!generate) log_error_detail(const_cast<char*>("error output")); end_log_file();
#endif
		exit(EXIT_FAILURE);
	} //mem allocate failure

	copy_cuda_memory<real_t, real_t_device>(d_OUTPUT, d_INPUT, INPUT,
			matrixSize, generate);

	//====================================
	for (int iteration = 0; iteration < iterations; iteration++) { //================== Global test loop

		if (!iteration && device_warmup)
			printf("First iteration: device warmup. Please wait...\n");

		// Timer...
		global_time = mysecond();

		cudaMemset(d_OUTPUT, 0, matrixSize * sizeof(real_t));

		if (verbose)
			printf(",");

		kernel_time = mysecond();
#ifdef LOGS
		if (iteration || !device_warmup)
		if (!generate) start_iteration();
#endif
		//================== Device computation, HMxM
		lud_cuda<real_t_device>(d_INPUT, k);

		checkCudaErrors (cudaPeekAtLastError());

checkCudaErrors		(cudaDeviceSynchronize());checkCudaErrors
		(cudaPeekAtLastError());

		//====================================
#ifdef LOGS
		if (iteration || !device_warmup)
		if (!generate) end_iteration();
#endif
		kernel_time = mysecond() - kernel_time;

		if (iteration || !device_warmup) {
			total_kernel_time += kernel_time;
			min_kernel_time = std::min(min_kernel_time, kernel_time);
			max_kernel_time = std::max(max_kernel_time, kernel_time);
		}

		if (iteration || !device_warmup)
			if (verbose)
				printf("Device kernel time for iteration %d: %.3fs\n",
						iteration, kernel_time);

		if (verbose)
			printf(",");

		// Timer...
		time = mysecond();

		//if (kernel_errors != 0) {
		checkCudaErrors(
				cudaMemcpy(OUTPUT, d_OUTPUT, matrixSize * sizeof(real_t_device),
						cudaMemcpyDeviceToHost));
		if (generate) {
//			write_gold_file<float>(INPUT, gold_matrix_path, k);
			write_gold_file<real_t>(OUTPUT, gold_matrix_path, k);
		} else if (iteration || !device_warmup) {
			//~ if (memcmp(A, GOLD, sizeof(float) * k*k)) {
			if (badass_memcmp<real_t>(GOLD, OUTPUT, matrixSize)) {
				char error_detail[150];
				int host_errors = 0;

				printf("!");

#pragma omp parallel for
				for (int i = 0; (i < k); i++) {
					for (int j = 0; (j < k); j++) {
						if (OUTPUT[i + k * j] != GOLD[i + k * j])
#pragma omp critical
								{
							double r = OUTPUT[i + k * j];
							double g = GOLD[i + k * j];
							snprintf(error_detail, 150,
									"p: [%d, %d], r: %1.16e, e: %1.16e", i, j,
									r, g);
							if (verbose && (host_errors < 10))
								printf("%s\n", error_detail);
#ifdef LOGS
							if (!generate) log_error_detail(error_detail);
#endif
							host_errors++;
							//ea++;
							//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + k * j], GOLD[i + k * j], t_ea);

						}
					}
				}
				if (host_errors != 0) {
					//================== To ensure there is no corrupted data on the inputs of the next iteration
					read_matrix_from_file<real_t>(INPUT, GOLD,
							input_matrix_path, gold_matrix_path, verbose,
							generate, k, fault_injection);
					copy_cuda_memory<real_t, real_t_device>(d_OUTPUT, d_INPUT,
							INPUT, matrixSize, generate);
				}
				//====================================
				// printf("numErrors:%d", host_errors);

#ifdef LOGS
				if (!generate) log_error_count(host_errors);
#endif
			}
		}

		//====================================
		printf(".");
		fflush(stdout);
		//}
		//====================================

		if (iteration || !device_warmup)
			if (verbose)
				printf("Gold check time for iteration %d: %.3fs\n", iteration,
						mysecond() - time);

		if (iteration || !device_warmup)
			if (verbose) {
				/////////// PERF
				double outputpersec = (double) matrixSize / kernel_time;
				printf("SIZE:%d OUTPUT/S:%f\n", k, outputpersec);
				///////////
			}

		if (iteration || !device_warmup)
			if (verbose)
				printf("Iteration #%d time: %.3fs\n\n\n", iteration,
						mysecond() - global_time);
		fflush(stdout);
	}
	double averageKernelTime = total_kernel_time
			/ (iterations - (device_warmup ? 1 : 0));
	printf(
			"\n-- END --\nTotal kernel time: %.3fs\nIterations: %d\nAverage kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n",
			total_kernel_time, iterations, averageKernelTime, min_kernel_time,
			max_kernel_time);

#ifdef LOGS
	if (!generate) end_log_file();
#endif

	//================== Release device memory
	cudaFree(d_INPUT);
	cudaFree(d_OUTPUT);
	//====================================
	free(INPUT);
	free(OUTPUT);
	free(GOLD);
}

int main(int argc, char* argv[]) {
//================== Test vars
	int device_warmup = 1;
	std::string precision = "float";

//================== Read test parameters
	int verbose, k, fault_injection, iterations, generate, matrixSize;
	char *input_matrix_path, *gold_matrix_path;

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

	if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
		getCmdLineArgumentString(argc, (const char **) argv, "input",
				&input_matrix_path);
	} else {
		input_matrix_path = new char[100];
		snprintf(input_matrix_path, 100, "slud_input_%i.matrix",
				(signed int) DEFAULT_INPUT_SIZE);
		printf("Using default input path: %s\n", input_matrix_path);
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "gold")) {
		getCmdLineArgumentString(argc, (const char **) argv, "gold",
				&gold_matrix_path);
	} else {
		gold_matrix_path = new char[100];
		snprintf(gold_matrix_path, 100, "slud_gold_%i.matrix", (signed int) k);
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
	} else {
		fault_injection = 0;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "no-warmup")) {
		device_warmup = 0;
		printf(
				"!! The first iteration may not reflect real timing information\n");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "generate")) {
		generate = 1;
		device_warmup = 0;
		iterations = 1;
		printf(
				"Will generate input if needed and GOLD.\nIterations setted to 1. no-warmup setted to false.\n");
	} else {
		generate = 0;
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "precision")) {
		char *tmp_precision;
		getCmdLineArgumentString(argc, (const char **) argv, "precision",
				&tmp_precision);
		precision = std::string(tmp_precision);
	}

//	if (precision == "float") {
//		test_lud_radiation<float, float>(matrixSize, verbose, generate, k,
//				fault_injection, iterations, device_warmup, input_matrix_path,
//				gold_matrix_path, precision);
//	} else if (precision == "double") {
//		test_lud_radiation<double, double>(matrixSize, verbose, generate, k,
//				fault_injection, iterations, device_warmup, input_matrix_path,
//				gold_matrix_path, precision);
//	} else
		if (precision == "half") {
		test_lud_radiation<half_float::half, half>(matrixSize, verbose,
				generate, k, fault_injection, iterations, device_warmup,
				input_matrix_path, gold_matrix_path, precision);
	}

	return 0;
}
