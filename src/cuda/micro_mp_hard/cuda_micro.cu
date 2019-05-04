#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <omp.h>
#include <random>
#include <cuda_fp16.h>
#include <vector>
#include <sstream>

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

// Returns the number of errors found
// if no errors were found it returns 0
template<typename T>
int check_output_errors(std::vector<T> &R, T OUTPUT_R, bool verbose) {
	int host_errors = 0;
	double gold = double(OUTPUT_R);
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < R.size(); i++) {
		double output = double(R[i]);
		if (gold != output) {
#pragma omp critical
			{
//					char error_detail[150];
//					snprintf(error_detail, 150, "p: [%d], r: %1.20e, e: %1.20e",
//							i, (double) output, (double) valGold);

				std::stringstream error_detail;
				error_detail << "p: [" << i << "], r: " << std::scientific
						<< output << ", e: " << gold;

				if (verbose && (host_errors < 10))
					std::cout << error_detail.str() << std::endl;
#ifdef LOGS
				log_error_detail(const_cast<char*>(error_detail.str().c_str()));
#endif
				host_errors++;
			}
		}
	}

	if (host_errors != 0) {
		std::cout << "#";
#ifdef LOGS
		log_error_count(host_errors);
#endif
	}
	return host_errors;
}

template<typename full, typename incomplete = void>
void test_radiation(const incomplete OUTPUT_R, const incomplete INPUT_A,
		const incomplete INPUT_B, Parameters& parameters) {
	// Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;
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

		//check output
		host_vector_full = device_vector_full.to_vector();
		int errors = check_output_errors<full>(host_vector_full, OUTPUT_R,
				parameters.verbose);
		unsigned long long relative_errors = copy_errors();

		if (parameters.verbose) {
			/////////// PERF
			double outputpersec = double(parameters.r_size) / kernel_time;
			std::cout << "SIZE:" << parameters.r_size;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " ITERATION " << iteration;
			std::cout << " time: " << kernel_time;
			std::cout << " output errors: " << errors;
			std::cout << " relative errors: " << relative_errors << std::endl;

		}
	}

	double averageKernelTime = total_kernel_time / parameters.iterations;
	std::cout << std::endl << "-- END --" << std::endl;
	std::cout << "Total kernel time: " << total_kernel_time << std::endl;
	std::cout << "Iterations: " << parameters.iterations << std::endl;
	std::cout << "Average kernel time: " << averageKernelTime << std::endl;
	std::cout << "Best: " << min_kernel_time << std::endl;
	std::cout << "Worst: " << max_kernel_time << std::endl;
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

	parameters.print_details();

	dmr(parameters);

#ifdef LOGS
	end_log_file();
#endif
	return 0;
}
