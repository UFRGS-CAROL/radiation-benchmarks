/*
 * nonconstant_setup.h
 *
 *  Created on: Jul 30, 2019
 *      Author: fernando
 */

#ifndef NONCONSTANT_SETUP_H_
#define NONCONSTANT_SETUP_H_
#include <fstream>
#include <iomanip>
#include <tuple>
#include <algorithm>
#include <cfloat>

#include "include/cuda_utils.h"
#include "include/device_vector.h"

#include "utils.h"
#include "dmr_nonconstant_kernels.h"

#ifdef LOGS
#include "log_helper.h"
#endif

#include "BinaryDouble.h"

void exception(std::string msg, std::string file, int line) {
	throw std::runtime_error(msg + " at " + file + ":" + std::to_string(line));
}

#define throw_line(msg) exception(msg, __FILE__, __LINE__)

bool cmp(const double lhs, const double rhs, const double zero) {
	const double diff = abs(lhs - rhs);
	if (diff > zero) {
		return false;
	}
	return true;
}

unsigned long long copy_errors() {
	unsigned long long errors_host = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&errors_host, errors,
					sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));

	unsigned long long temp = 0;
	//Reset the errors variable
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &temp, sizeof(unsigned long long), 0,
					cudaMemcpyHostToDevice));
	return errors_host;
}

template<typename real_t>
void load_file_data(std::string& path, std::vector<real_t>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
	}
	input.close();
}

template<typename real_t>
void write_to_file(std::string& path, std::vector<real_t>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
	}
	output.close();
}

template<typename real_t>
std::tuple<real_t, real_t, real_t, int> get_thresholds(
		std::vector<real_t> threshold_array) {
	real_t max_ = real_t(FLT_MIN_EXP);
	real_t min_ = real_t(FLT_MAX_EXP);
	int last_max_i;

	for (auto i = 0; i < threshold_array.size(); i++) {
		max_ = std::max(max_, threshold_array[i]);
		min_ = std::min(min_, threshold_array[i]);
		if (max_ == threshold_array[i]) {
			last_max_i = i;
		}
	}

	std::sort(threshold_array.begin(), threshold_array.end());

	real_t median = threshold_array[threshold_array.size() / 2];
	return std::make_tuple(max_, min_, median, last_max_i);
}

// Returns the number of errors found
// if no errors were found it returns 0
template<typename half_t, typename real_t>
int check_output_errors(std::vector<real_t> &output_real_t,
		std::vector<half_t> &output_half_t, std::vector<real_t>& gold_real_t,
		bool verbose, unsigned long long dmr_errors) {
	int host_errors = 0;

#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < output_real_t.size(); i++) {
		double output = double(output_real_t[i]);
		double output_inc = double(output_half_t[i]);
		double gold = double(gold_real_t[i]);

		if (output != gold || !cmp(output, output_inc, ZERO_FLOAT)) {
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

template<typename half_t, typename real_t>
void test_radiation(Parameters& parameters, std::vector<real_t>& input_array,
		std::vector<real_t>& gold_array) {
#ifdef CHECKBLOCK
	std::cout << "Instruction block checking size " << CHECKBLOCK << std::endl;
#endif
	// Init test environment
	// kernel_errors=0;
	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = 0;
	//====================================

	// real_t PRECIISON
	//output
	std::vector<real_t> output_host_vector_real_t(parameters.r_size, 0);
	rad::DeviceVector<real_t> output_device_vector_real_t(parameters.r_size);
	//Input
	rad::DeviceVector<real_t> input_device_vector_real_t(input_array);

	//threshold
	std::vector<real_t> threshold_host_real_t(parameters.r_size, 0);
	rad::DeviceVector<real_t> threshold_device_real_t(parameters.r_size);
	//====================================

	// SECOND PRECISION ONLY IF IT IS DEFINED
	std::vector<half_t> output_host_vector_half_t(parameters.r_size, 0);
	rad::DeviceVector<half_t> output_device_vector_half_t(parameters.r_size);

	//====================================
	// Verbose in csv format
	std::ofstream out(
			"./temp_" + parameters.instruction_str + "_"
					+ std::to_string(parameters.max_random) + ".csv");
	if (parameters.verbose == false) {
		out
				<< "output/s,iteration,time,output errors,max threshold,max output real_t, "
						"output half_t,threshold most significant bit, xor result"
				<< std::endl;
	}

	for (int iteration = 0; iteration < parameters.iterations; iteration++) {
		//================== Global test loop
		double kernel_time = rad::mysecond();
#ifdef LOGS
		start_iteration();
#endif
		//================== Device computation
		void (*micro_kernel)(real_t*, real_t*, real_t*, half_t*, int);
		if (parameters.redundancy == NONE) {
			throw_line("Not implemented");
		} else {
			switch (parameters.micro) {
			case ADD: {
				micro_kernel = &MicroBenchmarkKernel_ADDNONCONSTANT;
				break;
			}
			case MUL: {
				micro_kernel = &MicroBenchmarkKernel_MULNONCONSTANT;
				break;
			}
			case FMA: {
				micro_kernel = &MicroBenchmarkKernel_FMANONCONSTANT;
				break;
			}
			}
		}

		micro_kernel<<<parameters.grid_size, parameters.block_size>>>(
				input_device_vector_real_t.data(),	//input
				output_device_vector_real_t.data(), //output real
				threshold_device_real_t.data(),		// threshold half
				output_device_vector_half_t.data(), 		// output half
				parameters.operation_num);			//number of operations

		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		rad::checkFrameworkErrors(cudaPeekAtLastError());

		kernel_time = rad::mysecond() - kernel_time;

		//====================================
#ifdef LOGS
		end_iteration();
#endif

		total_kernel_time += kernel_time;
		min_kernel_time = std::min(min_kernel_time, kernel_time);
		max_kernel_time = std::max(max_kernel_time, kernel_time);

		//check output
		output_host_vector_real_t = output_device_vector_real_t.to_vector();
		output_host_vector_half_t = output_device_vector_half_t.to_vector();
		threshold_host_real_t = threshold_device_real_t.to_vector();

		real_t max_threshold, min_threshold, median, last_i;
		std::tie(max_threshold, min_threshold, median, last_i) = get_thresholds(
				threshold_host_real_t);

		unsigned long long relative_errors = copy_errors();
		int errors = 0;
		if (parameters.generate == false) {
			errors = check_output_errors(output_host_vector_real_t,
					output_host_vector_half_t, gold_array, parameters.verbose,
					relative_errors);
		}

		double outputpersec = double(parameters.r_size) / kernel_time;
		std::cout << std::scientific << std::setprecision(20);
		BinaryDouble bd = max_threshold;

		if (parameters.verbose) {
			/////////// PERF
			std::cout << "-----------------------------------------------"
					<< std::endl;

			std::cout << "ITERATION " << iteration << std::endl;
			std::cout << "SIZE:" << parameters.r_size << std::endl;
			std::cout << "OUTPUT/S:" << outputpersec << std::endl;
			std::cout << "TIME: " << kernel_time << std::endl;
			std::cout << "OUTPUT ERRORS: " << errors << std::endl;
			std::cout << "RELATIVE ERRORS: " << relative_errors << std::endl;
			std::cout << "MAX THRESHOLD: " << max_threshold << std::endl;
			std::cout << "MIN THRESHOLD: " << min_threshold << std::endl;
			std::cout << "MEDIAN THRESHOLD: " << median << std::endl;
			std::cout << std::setprecision(0) << std::fixed;
			std::cout << "MOST SIGNIFICANT biT: " << bd.most_significant_bit()
					<< std::endl;
			std::cout << "MAX BINARY: " << bd << std::endl;
			std::cout << "input[" << last_i << "] for MAX THRESHOLD: ";
			std::cout << std::scientific << std::setprecision(20)
					<< input_array[last_i] << std::endl;

			std::cout << "-----------------------------------------------"
					<< std::endl;

		} else {
			BinaryDouble biggest_threshold_output_real_t =
					output_host_vector_real_t[last_i];
			BinaryDouble biggest_threshold_output_half_t =
					output_host_vector_half_t[last_i];
			BinaryDouble xor_result = biggest_threshold_output_real_t
					^ biggest_threshold_output_half_t;
			// CSV format
			out << outputpersec << ",";
			out << iteration << ",";
			out << kernel_time << ",";
			out << errors << ",";
			out << max_threshold << ",";
			out << output_host_vector_real_t[last_i] << ",";
			out << output_host_vector_half_t[last_i] << ",";
			out << xor_result.most_significant_bit() << ",";
			out << xor_result << std::endl;
		}
	}

	out.close();
	if (parameters.verbose) {
		double averageKernelTime = total_kernel_time / parameters.iterations;
		std::cout << std::endl << "-- END --" << std::endl;
		std::cout << "Total kernel time: " << total_kernel_time << std::endl;
		std::cout << "Iterations: " << parameters.iterations << std::endl;
		std::cout << "Average kernel time: " << averageKernelTime << std::endl;
		std::cout << "Best: " << min_kernel_time << std::endl;
		std::cout << "Worst: " << max_kernel_time << std::endl;
	}

	if (parameters.generate) {
		gold_array = output_host_vector_real_t;
	}
}

template<typename real_t>
void generate_input_file(Parameters& parameters) {
	std::vector<real_t> input(parameters.r_size);
	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<double> dis(parameters.min_random,
			parameters.max_random);

	for (auto& in : input) {
		in = real_t(dis(gen));
	}
	write_to_file(parameters.input_file, input);
}

template<typename half_t, typename real_t = half_t>
void setup(Parameters& parameters) {
	std::vector<real_t> input_array(parameters.r_size);
	std::vector<real_t> gold_array(parameters.r_size);

	if (parameters.generate) {
		generate_input_file<real_t>(parameters);
	} else {
		load_file_data(parameters.gold_file, gold_array);
	}
	load_file_data(parameters.input_file, input_array);

	test_radiation<half_t, real_t>(parameters, input_array, gold_array);

	if (parameters.generate) {
		write_to_file(parameters.gold_file, gold_array);
	}
}

void dmr_nonconstant(Parameters& parameters) {
	/* DMRMIXED REDUNDANCY -------------------------------------------------- */
	if (parameters.redundancy == DMRMIXED) {
		if (parameters.precision == DOUBLE) {
			setup<float, double>(parameters);
		}

		if (parameters.precision == SINGLE) {
			throw_line("Single/Half not implemented!");
		}
	}

	/* DMR REDUNDANCY ------------------------------------------------------- */
	/* NONE REDUNDANCY ------------------------------------------------------ */
	if (parameters.redundancy == NONE || parameters.redundancy == DMR) {
		if (parameters.precision == HALF) {
			throw_line("Half version not implemented!");
		}

		if (parameters.precision == SINGLE) {
			setup<float>(parameters);
		}

		if (parameters.precision == DOUBLE) {
			setup<double>(parameters);
		}
	}
}

#endif /* NONCONSTANT_SETUP_H_ */
