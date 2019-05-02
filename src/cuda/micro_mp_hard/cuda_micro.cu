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
#include "none_kernels.h"
#include "device_vector.h"
#include "cuda_utils.h"
#include "Parameters.h"

// helper functions
#include "helper_cuda.h"

#define HALF_ROUND_STYLE 1
#define HALF_ROUND_TIES_TO_EVEN 1
#include "half.hpp"

#define BLOCK_SIZE 32

#define DEFAULT_INPUT_SIZE 8192

void usage(int argc, char* argv[]) {
	printf("Usage: %s [-iterations=N] [-verbose]\n", argv[0]);
}

// Returns true if no errors are found. False if otherwise.
// Set votedOutput pointer to retrieve the voted matrix
template<typename T>
int check_output_errors(std::vector<T> &R, T OUTPUT_R, bool verbose) {
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
	return host_errors;
}

template<typename full, typename incomplete = void>
void test_radiation(const incomplete OUTPUT_R, const incomplete INPUT_A,
		const incomplete INPUT_B, Parameters& parameters) {
	//================== Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;

	parameters.print_details();
	//====================================
	// FULL PRECIISON
	std::vector<full> host_vector_full(parameters.r_size, 0);
	DeviceVector<full> device_vector_full(parameters.r_size);

	//====================================
	// SECOND PRECISION ONLY IF IT IS DEFINED
	DeviceVector<incomplete> device_vector_inc;
	std::vector<incomplete> host_vector_inc;
	if (std::is_void<incomplete>::value != true) {
		host_vector_inc = std::vector<incomplete>(parameters.r_size, 0);
		device_vector_inc = DeviceVector<incomplete>(parameters.r_size);
	}

	for (int iteration = 0; iteration < parameters.iterations; iteration++) {
		//================== Global test loop
		double kernel_time = mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
		if (std::is_void<incomplete>::value) {
			switch (parameters.micro) {
			case ADD:
				MicroBenchmarkKernel_ADD<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			case MUL:
				MicroBenchmarkKernel_MUL<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			case FMA:
				MicroBenchmarkKernel_FMA<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			}
		} else {
			switch (parameters.micro) {
			case ADD:
				MicroBenchmarkKernel_ADD<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data, device_vector_full.data, 0.1,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			case MUL:
				MicroBenchmarkKernel_MUL<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data, device_vector_full.data, 0.1,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			case FMA:
				MicroBenchmarkKernel_FMA<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data, device_vector_full.data, 0.1,
						OUTPUT_R, INPUT_A, INPUT_B);
				break;
			}
		}

		checkFrameworkErrors(cudaPeekAtLastError());
		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());

		kernel_time = mysecond() - kernel_time;

		//====================================
#ifdef LOGS
		end_iteration();
#endif

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

		std::cout << ".";
		if (parameters.verbose) {
			//check output
			host_vector_full = device_vector_full.to_vector();
			int errors = check_output_errors<full>(host_vector_full, OUTPUT_R,
					parameters.verbose);
			unsigned long long relative_errors = copy_errors();

			/////////// PERF
			double outputpersec = double(parameters.r_size) / kernel_time;
			std::cout << "SIZE:" << parameters.r_size;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " ITERATION " << iteration << " time: " << kernel_time
					<< std::endl;

		}
	}

	double gflops = parameters.r_size * OPS / 1e9; // Billion FLoating-point OPerationS
	double averageKernelTime = total_kernel_time / parameters.iterations;
	std::printf("\n-- END --\n"
			"Total kernel time: %.3fs\n"
			"Iterations: %d\n"
			"Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n"
			"Average GFLOPs: %.2f (best: %.2f ; worst: %.2f)\n",
			total_kernel_time, parameters.iterations, averageKernelTime,
			min_kernel_time, max_kernel_time, gflops / averageKernelTime,
			gflops / min_kernel_time, gflops / max_kernel_time);

}

void dmr(Parameters& parameters) {
	switch (parameters.redundancy) {
	//NONE REDUNDANCY ----------------------------------------------------------
	case NONE:
		switch (parameters.precision) {
		case HALF:
			test_radiation<half>(OUTPUT_R_HALF, INPUT_A_HALF, INPUT_B_HALF,
					parameters);
			break;
		case SINGLE:
			test_radiation<float>(OUTPUT_R_SINGLE, INPUT_A_SINGLE,
			INPUT_B_SINGLE, parameters);
			break;

		case DOUBLE:
			test_radiation<double>(OUTPUT_R_DOUBLE, INPUT_A_DOUBLE,
			INPUT_B_DOUBLE, parameters);
			break;
		}
		break;

		//DMR MIXED REDUNDANCY -------------------------------------------------------
	case DMRMIXED:
		switch (parameters.precision) {
		case DOUBLE:
			test_radiation<double, float>(OUTPUT_R_SINGLE, INPUT_A_SINGLE,
			INPUT_B_SINGLE, parameters);
			break;
		case SINGLE:
			test_radiation<float, half>(OUTPUT_R_HALF, INPUT_A_HALF,
			INPUT_B_HALF, parameters);
			break;
		}
		break;

//		//DMR REDUNDANCY -------------------------------------------------------
	case DMR:
		switch (parameters.precision) {
		case DOUBLE:
			test_radiation<double, double>(OUTPUT_R_DOUBLE, INPUT_A_DOUBLE,
			INPUT_B_DOUBLE, parameters);
			break;
		case SINGLE:
			test_radiation<float, float>(OUTPUT_R_SINGLE, INPUT_A_SINGLE,
			INPUT_B_SINGLE, parameters);
			break;
		case HALF:
			test_radiation<half, half>(OUTPUT_R_HALF, INPUT_A_HALF,
			INPUT_B_HALF, parameters);
			break;
		}
		break;

	}

}

int main(int argc, char* argv[]) {

//================== Set block and grid size for MxM kernel
	cudaDeviceProp prop = GetDevice();
	Parameters parameters(argc, argv, prop.multiProcessorCount, 256);
//================== Init logs
#ifdef LOGS
	std::string test_info = std::string("ops:") + std::to_string(OPS)
	+ " gridsize:" + std::to_string(parameters.grid_size)
	+ " blocksize:" + std::to_string(parameters.block_size) + " type:"
	+ parameters.instruction_str + "-" + parameters.precision_str
	+ "-precision hard:" + parameters.hardening_str;

	std::string test_name = std::string("cuda_") + parameters.precision_str
	+ "_micro-" + parameters.instruction_str;
	start_log_file(const_cast<char*>(test_name.c_str()),
			const_cast<char*>(test_info.c_str()));
#endif

	dmr(parameters);

#ifdef LOGS
	end_log_file();
#endif
	return 0;
}
