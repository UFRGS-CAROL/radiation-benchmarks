#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <omp.h>
#include <random>
#include <cuda_fp16.h>
#include <vector>

#include "dmr_kernels.h"
#include "device_vector.h"
#include "cuda_utils.h"

// helper functions
#include "helper_string.h"
#include "helper_cuda.h"

#define HALF_ROUND_STYLE 1
#define HALF_ROUND_TIES_TO_EVEN 1
#include "half.hpp"

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

//===================================== DEFINE TESTED PRECISION
//FOR DMR APPROACH I NEED to use the smallest precision
//as a limit, since it is not possible to store the bigger precisions
//on smaller precisions

//If double it means that DMR will be double and float
//so the limits are the float ones

#define INPUT_A_DOUBLE 1.1945305291614955E+103 // 0x5555555555555555
#define INPUT_B_DOUBLE 3.7206620809969885E-103 // 0x2AAAAAAAAAAAAAAA
#define OUTPUT_R_DOUBLE 4.444444444444444 //0x4011C71C71C71C71

#define INPUT_A_SINGLE 1.4660155E+13 // 0x55555555
#define INPUT_B_SINGLE 3.0316488E-13 // 0x2AAAAAAA
#define OUTPUT_R_SINGLE 4.444444 //0x408E38E3

#define INPUT_A_HALF 1.066E+2 // 0x56AA
#define INPUT_B_HALF 4.166E-2 // 0x2955
#define OUTPUT_R_HALF 4.44 // 0x4471

#define OPS_PER_THREAD_OPERATION 1

//=====================================================

#ifdef FMA
const char test_type_description[] = "fma_dmr";
#endif

#ifdef ADD
const char test_type_description[] = "add_dmr";
#endif

#ifdef MUL
const char test_type_description[] = "mul_dmr";
#endif

#ifdef HALF
const char test_precision_description[] = "half";
#endif

#ifdef SINGLE
const char test_precision_description[] = "single";
#endif

#ifdef DOUBLE
const char test_precision_description[] = "double";
#endif

void usage(int argc, char* argv[]) {
	printf("Usage: %s [-iterations=N] [-verbose]\n", argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
template<typename T, int OUTPUT_R>
bool checkOutputErrors(std::vector<T> &R, bool verbose) {
	int host_errors = 0;
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < R.size(); i++) {
		register bool checkFlag = true;
		register T valGold = (OUTPUT_R);
		register T valOutput = R[i];
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

	if (host_errors != 0) {
		printf("#");
#ifdef LOGS
		log_error_count(host_errors);
#endif
	}
	return host_errors == 0;
}

template<typename incomplete, typename full>
void test_radiation(int iterations, bool verbose, int r_size, int gridsize,
		int blocksize) {
	//================== Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;
	double global_time;

	std::printf("cuda_micro-%s_%s\n", test_type_description, test_precision_description);
	//====================================
	std::vector<incomplete> host_vector_inc(r_size, 0);
	std::vector<full> host_vector_ful(r_size, 0);

	DeviceVector<incomplete> device_vector_inc;
	DeviceVector<full> device_vector_ful;

	device_vector_ful = host_vector_ful;
	device_vector_inc = host_vector_inc;

	for (int loop2 = 0; loop2 < iterations; loop2++) {
		//================== Global test loop

		global_time = mysecond();
		double kernel_time = mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
//#ifdef FMA
		MicroBenchmarkKernel_FMA_DMR<incomplete, full> <<<gridsize, blocksize>>>
				(device_vector_inc.data, device_vector_ful.data, 0.1,  OUTPUT_R_HALF, INPUT_A_HALF, INPUT_B_HALF);
//#endif

//#ifdef ADD
//		MicroBenchmarkKernel_ADD_DMR<<<gridsize, blocksize>>>(d_R);
//#endif
//
//#ifdef MUL
//		MicroBenchmarkKernel_MUL_DMR<<<gridsize, blocksize>>>(d_R);
//#endif
		checkFrameworkErrors(cudaPeekAtLastError());
		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());

		//====================================
#ifdef LOGS
		end_iteration();
#endif
		kernel_time = mysecond() - kernel_time;

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

		if (verbose)
			std::printf("Device kernel time for iteration %d: %.3fs\n", loop2,
					kernel_time);

		double gold_check_time = mysecond();

		std::printf(".");

		if (verbose)
			std::printf("Gold check time for iteration %d: %.3fs\n", loop2,
					mysecond() - gold_check_time);

		if (verbose) {
			/////////// PERF
			double flops = r_size * OPS * OPS_PER_THREAD_OPERATION;
			double gflops = flops / kernel_time;
			double outputpersec = (double) r_size / kernel_time;
			std::printf("SIZE:%d OUTPUT/S:%f FLOPS:%f (GFLOPS:%.2f)\n", r_size,
					outputpersec, gflops, gflops / 1000000000);
			///////////
		}

		if (verbose)
			std::printf("Iteration #%d time: %.3fs\n\n\n", loop2,
					mysecond() - global_time);
	}

	double gflops = r_size * OPS * OPS_PER_THREAD_OPERATION / 1000000000; // Bilion FLoating-point OPerationS
	double averageKernelTime = total_kernel_time / iterations;
	std::printf("\n-- END --\n"
			"Total kernel time: %.3fs\n"
			"Iterations: %d\n"
			"Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n"
			"Average GFLOPs: %.2f (best: %.2f ; worst: %.2f)\n",
			total_kernel_time, iterations, averageKernelTime, min_kernel_time,
			max_kernel_time, gflops / averageKernelTime,
			gflops / min_kernel_time, gflops / max_kernel_time);

#ifdef LOGS
	end_log_file();
#endif
}

int main(int argc, char* argv[]) {
	int iterations, verbose;

//================== Read test parameters
	if (checkCmdLineFlag(argc, (const char **) argv, "iterations")) {
		iterations = getCmdLineArgumentInt(argc, (const char **) argv,
				"iterations");
	}

	if (checkCmdLineFlag(argc, (const char **) argv, "verbose")) {
		verbose = 1;
	}
//================== Set block and grid size for MxM kernel
	cudaDeviceProp prop = GetDevice();
	int gridsize = prop.multiProcessorCount;
	int blocksize = 256;

	std::printf("grid size = %d ; block size = %d\n", gridsize, blocksize);

	int r_size = gridsize * blocksize * OPS_PER_THREAD_OPERATION;
//====================================

//================== Init logs
#ifdef LOGS
	char test_info[250];
	char test_name[250];
	snprintf(test_info, 250, "ops:%d gridsize:%d blocksize:%d type:%s-%s-precision", OPS, gridsize, blocksize, test_type_description, test_precision_description);
	snprintf(test_name, 250, "cuda_%s_micro-%s", test_precision_description, test_type_description);
	start_log_file(test_name, test_info);
#endif

	test_radiation<float, double>(iterations, verbose, r_size, gridsize,
			blocksize);
	return 0;
}
