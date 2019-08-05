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

bool cmp(const double lhs, const double rhs, const double t) {

	BinaryDouble rhs_ = rhs;
	BinaryDouble lhs_ = lhs;
	BinaryDouble test = (rhs_ ^ lhs_);

	return (test.most_significant_bit() < MAX_VALUE);
}

bool is_bigger_than_threshold(const float lhs, const double rhs) {
	float rhs_float = float(rhs);

	uint32* lhs_ptr = (uint32*) &lhs;
	uint32* rhs_ptr = (uint32*) &rhs_float;

	uint32 lhs_int = *lhs_ptr;
	uint32 rhs_int = *rhs_ptr;

	uint32 sub_res =
			(lhs_int > rhs_int) ? lhs_int - rhs_int : rhs_int - lhs_int;

	return (sub_res > MAX_VALUE);
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

/**
 * a gente pega o FP64, vai pra FP32
 a gente pega os dois 32 bit e considera como INT
 a gente faz SUB
 unsigned
 que a ultima coisa que queremos e ligar com sinal agora
 esse valor vai ser a thresold
 o seja, na injecao tu faz o mesmo, FP64 -> FP32
 uint(FP32) - unit(FP32) > threshold?
 sim -> erro
 não -> de boa
 */
template<typename half_t, typename real_t, typename int_t>
std::tuple<int_t, int_t, int_t, int_t, int_t> get_thresholds(
		std::vector<half_t>& half_array, std::vector<real_t>& real_array) {

	assert(sizeof(int_t) == sizeof(half_t));
	std::vector<int_t> xor_array(real_array.size());

	int_t min_ = 0xffffffff;
	int_t max_ = 0;
	int_t max_i = 0;
	int_t min_i = 0;
	for (int i = 0; i < real_array.size(); i++) {
		half_t output_half_t_float = half_t(half_array[i]);
		half_t output_real_t_float = half_t(real_array[i]);

		int_t* lhs_ptr = (int_t*)&output_half_t_float;
		int_t* rhs_ptr = (int_t*)&output_real_t_float;

		int_t lhs_int = *lhs_ptr;
		int_t rhs_int = *rhs_ptr;

		int_t most_significant =
				(lhs_int > rhs_int) ? lhs_int - rhs_int : rhs_int - lhs_int;


		min_ = std::min(most_significant, min_);
		max_ = std::max(most_significant, max_);
		if (min_ == most_significant) {
			min_i = i;
		}

		if (max_ == most_significant) {
			max_i = i;
		}
	}

	std::sort(xor_array.begin(), xor_array.end(), std::greater<int_t>());

	int_t median = xor_array[xor_array.size() / 2];
	return std::make_tuple(max_, min_, median, max_i,
			min_i);
}

// Returns the number of errors found
// if no errors were found it returns 0
template<typename half_t, typename real_t>
int check_output_errors(std::vector<real_t> &output_real_t,
		std::vector<half_t> &output_half_t, std::vector<real_t>& gold_real_t,
		bool verbose, unsigned long long dmr_errors) {
	int host_errors = 0;
	unsigned dmr_int_error = 0;
#pragma omp parallel for shared(host_errors)
	for (int i = 0; i < output_real_t.size(); i++) {
		double output = double(output_real_t[i]);
		double output_inc = double(output_half_t[i]);
		double gold = double(gold_real_t[i]);
		bool cmp_dmr = is_bigger_than_threshold(output_half_t[i], gold);
		dmr_int_error += cmp_dmr;
		if (output != gold || cmp_dmr) {
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

	if (dmr_errors != 0 || dmr_int_error) {
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
			"./temp_" + std::to_string(parameters.operation_num) + "_"
					+ parameters.instruction_str + "_"
					+ std::to_string(parameters.min_random) + "_"
					+ std::to_string(parameters.max_random) + "_" + ".csv",
			std::ios::out);
	if (parameters.verbose == false) {
		out
				<< "output/s,iteration,time,output_errors,max_threshold,max_real,max_half,min_threshold,"
						"min_real,min_half,median_threshold,median_real,median_half"
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

		uint32 max_threshold, min_threshold, median, min_i, max_i;
		std::tie(max_threshold, min_threshold, median, max_i, min_i) =
				get_thresholds<half_t, real_t, uint32>(
						output_host_vector_half_t, output_host_vector_real_t);

		unsigned long long relative_errors = copy_errors();
		int errors = 0;
		if (parameters.generate == false) {
			errors = check_output_errors(output_host_vector_real_t,
					output_host_vector_half_t, gold_array, parameters.verbose,
					relative_errors);
		}

		double outputpersec = double(parameters.r_size) / kernel_time;
		std::cout << std::scientific << std::setprecision(20);
		out << std::scientific << std::setprecision(20);

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
			std::cout << "MOST SIGNIFICANT biT: " << max_threshold << std::endl;
//			std::cout << "MAX BINARY: " << BinaryDouble(max_threshold)
//					<< std::endl;
//			std::cout << "input[" << last_i << "] for MAX THRESHOLD: ";
//			std::cout << std::scientific << std::setprecision(20)
//					<< input_array[last_i] << std::endl;

			std::cout << "-----------------------------------------------"
					<< std::endl;

		} else {
			// CSV format
			out << outputpersec << ",";
			out << iteration << ",";
			out << kernel_time << ",";
			out << errors << ",";

			out << max_threshold << ",";
			out << output_host_vector_real_t[max_i] << ",";
			out << output_host_vector_half_t[max_i] << ",";

			out << min_threshold << ",";
			out << output_host_vector_real_t[min_i] << ",";
			out << output_host_vector_half_t[min_i] << ",";

			out << median << ",";
			out
					<< output_host_vector_real_t[output_host_vector_half_t.size()
							/ 2] << ",";
			out
					<< output_host_vector_half_t[output_host_vector_half_t.size()
							/ 2];

//			out << xor_result.most_significant_bit() << ",";
//			out << xor_result << std::endl;
			out << std::endl;
		}
	}

	out.close();
	if (parameters.verbose) {
		std::cout << std::scientific << std::setprecision(20);

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
