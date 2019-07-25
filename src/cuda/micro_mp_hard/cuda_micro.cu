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
#include "include/device_vector.h"
#include "cuda_utils.h"
#include "Parameters.h"

// helper functions
#include "helper_cuda.h"

//#define HALF_ROUND_STYLE 1
//#define HALF_ROUND_TIES_TO_EVEN 1
//#include "half.hpp"

std::string get_double_representation(double val) {
	std::string output = "";
	if (sizeof(double) == 8) {

		uint64_t int_val;

		memcpy(&int_val, &val, sizeof(double));
		for (uint64_t i = uint64_t(1) << 63; i > 0; i = i / 2) {
			if (int_val & i) {
				output += "1";
			} else {
				output += "0";
			}
		}
	} else {
		std::cerr << "USING more than 64 bits double" << std::endl;
	}
	return output;
}

bool cmp(const double lhs, const double rhs, const double zero) {
	const double diff = abs(lhs - rhs);
	if (diff > zero) {
		return false;
	}
	return true;
}

// Returns the number of errors found
// if no errors were found it returns 0
template<typename incomplete, typename full, typename output_type = incomplete>
int check_output_errors(std::vector<incomplete> &R_incomplete,
		std::vector<full> &R, output_type OUTPUT_R, bool verbose,
		unsigned long long dmr_errors) {
	int host_errors = 0;
	double gold = double(OUTPUT_R);
	double threshold = -3;
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < R.size(); i++) {
		double output = double(R[i]);
		double output_inc = double(R_incomplete[i]);
		threshold = max(threshold, fabs(output - output_inc));
		if (!cmp(gold, output, 0.000000001)
				|| !cmp(output, output_inc, ZERO_FLOAT)) {
#pragma omp critical
			{
				std::stringstream error_detail;
				error_detail.precision(20);
				error_detail << "p: [" << i << "], r: " << std::scientific
						<< output << ", e: " << gold << " smaller_precision: "
						<< output_inc;

				if (verbose && (host_errors < 10))
					std::cout << error_detail.str() << std::endl;
#ifdef LOGS
				log_error_detail(const_cast<char*>(error_detail.str().c_str()));
#endif
				host_errors++;
			}
		}
	}

	if (dmr_errors != 0) {
		std::stringstream error_detail;
		error_detail << "detected_dmr_errors: " << dmr_errors;
		;
#ifdef LOGS
		log_error_detail(const_cast<char*>(error_detail.str().c_str()));
#endif
	}

	if (host_errors != 0) {
		std::cout << "#";
#ifdef LOGS
		log_error_count(host_errors);
#endif
	}
	return host_errors;
}

template<typename incomplete, typename full, typename ... TypeArgs>
void test_radiation(Type<TypeArgs...>& type_, Parameters& parameters) {
	std::cout << "Input values " << type_ << std::endl;
#ifdef CHECKBLOCK
	std::cout << "Instruction block checking size " << CHECKBLOCK << std::endl;
#endif
	// Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;
	//====================================

	// FULL PRECIISON
	std::vector<full> host_vector_full(parameters.r_size, 0);
	rad::DeviceVector<full> device_vector_full(parameters.r_size);

	//====================================

	// SECOND PRECISION ONLY IF IT IS DEFINED
	std::vector<incomplete> host_vector_inc(parameters.r_size, 0);
	rad::DeviceVector<incomplete> device_vector_inc(parameters.r_size);
	//====================================
	// Verbose in csv format
	if (parameters.verbose == false) {
		std::cout << "output/s,iteration,time,output errors,relative errors"
				<< std::endl;
	}

	auto gold = type_.output_r;
	for (int iteration = 0; iteration < parameters.iterations; iteration++) {
		//================== Global test loop
		double kernel_time = mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
		if (parameters.redundancy == NONE) {
			switch (parameters.micro) {
			case ADD:
				MicroBenchmarkKernel_ADD<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data(),
						type_.output_r, type_.input_a);
				break;
			case MUL:
				MicroBenchmarkKernel_MUL<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data(),
						type_.output_r, type_.input_a);
				break;
			case FMA:
				MicroBenchmarkKernel_FMA<full> <<<parameters.grid_size,
						parameters.block_size>>>(device_vector_full.data(),
						type_.output_r, type_.input_a, type_.input_b);
				break;
			}

		} else {
			switch (parameters.micro) {
			case ADD: {
				MicroBenchmarkKernel_ADD<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r, type_.input_a);
				break;
			}
			case MUL: {
				MicroBenchmarkKernel_MUL<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r, type_.input_a);
				break;
			}
			case FMA: {
				MicroBenchmarkKernel_FMA<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r, type_.input_a, type_.input_b);
				break;
			}
			case ADDNOTBIASED: {
				MicroBenchmarkKernel_ADDNOTBIASAED<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r);
				break;
			}
			case MULNOTBIASED: {
				MicroBenchmarkKernel_MULNOTBIASAED<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r);
				gold = 1.10517102313140469505;
				break;
			}
			case FMANOTBIASED: {
				MicroBenchmarkKernel_FMANOTBIASAED<incomplete, full> <<<
						parameters.grid_size, parameters.block_size>>>(
						device_vector_inc.data(), device_vector_full.data(),
						type_.output_r);
				gold = 2.00324498477763457416e+00;
				break;
			}
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

		//check output
		host_vector_full = device_vector_full.to_vector();
		host_vector_inc = device_vector_inc.to_vector();
		unsigned long long relative_errors = copy_errors();

		int errors = check_output_errors(host_vector_inc, host_vector_full,
				gold, parameters.verbose, relative_errors);

		double outputpersec = double(parameters.r_size) / kernel_time;
		if (parameters.verbose) {
			/////////// PERF
			std::cout << "SIZE:" << parameters.r_size;
			std::cout << " OUTPUT/S:" << outputpersec;
			std::cout << " ITERATION " << iteration;
			std::cout << " time: " << kernel_time;
			std::cout << " output errors: " << errors;
			std::cout << " relative errors: " << relative_errors << std::endl;

		} else {
			// CSV format
			std::cout << outputpersec << ",";
			std::cout << iteration << ",";
			std::cout << kernel_time << ",";
			std::cout << errors << ",";
			std::cout << relative_errors << std::endl;

		}
	}

	if (parameters.verbose) {
		double averageKernelTime = total_kernel_time / parameters.iterations;
		std::cout << std::endl << "-- END --" << std::endl;
		std::cout << "Total kernel time: " << total_kernel_time << std::endl;
		std::cout << "Iterations: " << parameters.iterations << std::endl;
		std::cout << "Average kernel time: " << averageKernelTime << std::endl;
		std::cout << "Best: " << min_kernel_time << std::endl;
		std::cout << "Worst: " << max_kernel_time << std::endl;
	}
}

void dmr(Parameters& parameters) {
	/* DMRMIXED REDUNDANCY -------------------------------------------------- */
	if (parameters.redundancy == DMRMIXED) {

		if (parameters.precision == DOUBLE) {
			Type<float, double> type_;
//			Type<float> type_;
			test_radiation<float, double, float, double>(type_, parameters);
//			test_radiation<float, double>(type_, parameters);

		}

		if (parameters.precision == SINGLE) {
//			Type<half, float> type_;
			Type<half> type_;
//			test_radiation<half, float, half, float>(type_, parameters);
			test_radiation<half, float>(type_, parameters);

		}
	}

	/* DMR REDUNDANCY ------------------------------------------------------- */
	/* NONE REDUNDANCY ------------------------------------------------------ */
	if (parameters.redundancy == NONE || parameters.redundancy == DMR) {
		if (parameters.precision == HALF) {
			Type<half> type_;
			test_radiation<half, half>(type_, parameters);

		}

		if (parameters.precision == SINGLE) {
			Type<float> type_;
			test_radiation<float, float>(type_, parameters);
		}

		if (parameters.precision == DOUBLE) {
			Type<double> type_;
			test_radiation<double, double>(type_, parameters);
		}

	}

}

int main(int argc, char* argv[]) {

//================== Set block and grid size for MxM kernel
	cudaDeviceProp prop = GetDevice();
	Parameters parameters(argc, argv, prop.multiProcessorCount, 256);
	if (parameters.verbose) {
		std::cout << "Get device Name: " << prop.name << std::endl;
	}
//================== Init logs
#ifdef LOGS
	std::string test_info = std::string("ops:") + std::to_string(OPS)
	+ " gridsize:" + std::to_string(parameters.grid_size)
	+ " blocksize:" + std::to_string(parameters.block_size) + " type:"
	+ parameters.instruction_str + "-" + parameters.precision_str
	+ "-precision hard:" + parameters.hardening_str;
	test_info += " checkblock:";
#ifdef CHECKBLOCK
	test_info += std::to_string(CHECKBLOCK);
#else
	test_info += std::to_string(OPS);
#endif

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
