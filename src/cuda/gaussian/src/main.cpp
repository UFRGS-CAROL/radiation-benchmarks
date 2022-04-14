#include <vector>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "cuda_utils.h"
#include "generic_log.h"
#include "Parameters.h"
#include "utils.h"

extern std::string get_multi_compiler_header();

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
/**
 * Benchmark from rodinia benchmark suite,
 *  updated for radiation benchmarks by Fernando (fernandofernandesantos@gmail.com) 2020
 */

template<typename real_t>
void create_matrix(std::vector<real_t>& m, size_t size) {
	real_t lamda = -0.01;
	std::vector<real_t> coe(2 * size - 1);
	real_t coe_i = 0.0;

	for (size_t i = 0; i < size; i++) {
		coe_i = 10 * std::exp(lamda * i);
		auto j = size - 1 + i;
		coe[j] = coe_i;
		j = size - 1 - i;
		coe[j] = coe_i;
	}

	for (size_t i = 0; i < size; i++) {
		for (size_t j = 0; j < size; j++) {
			m[i * size + j] = coe[size - 1 - i + j];
		}
	}

}

template<typename real_t>
bool read_from_file(std::string& path, std::vector<real_t>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
		input.close();
		return false;
	}
	return true;
}

template<typename real_t>
bool write_to_file(std::string& path, std::vector<real_t>& array) {
	std::ofstream output(path, std::ios::binary);
	if (output.good()) {
		output.write(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
		output.close();

		return false;
	}
	return true;
}

template<typename real_t>
size_t check_and_log(std::vector<real_t>& final_vector,
		std::vector<real_t>& gold_final_vector, rad::Log& log, bool verbose) {
	size_t error_count = 0;
#pragma omp parallel for default(shared)
	for (size_t i = 0; i < gold_final_vector.size(); i++) {
		auto g = gold_final_vector[i];
		auto f = final_vector[i];
		if (g != f) {
			std::stringstream error_detail;
			error_detail << std::setprecision(20) << std::scientific;
			//#ERR m: [870, 853], r: -2.2781341075897217e+00, e: -3.3776178359985352e+00
			error_detail << "m: [" << i << "], r: " << f << ", e: " << g;
#pragma omp critical
			{
				log.log_error_detail(error_detail.str());
				error_count++;
				if (verbose) {
					std::cout << error_detail.str() << std::endl;
				}
			}
		}
	}
	log.update_errors();
	return error_count;
}

int main(int argc, char *argv[]) {

	Parameters parameters(argc, argv);
	std::string test_name = "cudaGaussian";
	std::string test_info = "size:" + std::to_string(parameters.size);
	test_info += " maxblocksize:" + std::to_string(MAXBLOCKSIZE);
	test_info += " blocksizexy:" + std::to_string(BLOCK_SIZE_XY);
	test_info += " " + get_multi_compiler_header();

	rad::Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << log << std::endl;
		std::cout << "WG size of kernel 1 = " << MAXBLOCKSIZE
				<< ", WG size of kernel 2= " << BLOCK_SIZE_XY << " X "
				<< BLOCK_SIZE_XY << std::endl;
	}

	size_t matrix_size = parameters.size * parameters.size;
	std::vector<float> a_host(matrix_size);
	std::vector<float> m_host(matrix_size, 0.0f);

	std::vector<float> b_host(parameters.size, 1.0f);
	std::vector<float> final_vector(parameters.size);
	std::vector<float> gold_final_vector(parameters.size);
	std::vector<float> a_copy_host(a_host);
	std::vector<float> b_copy_host(b_host);
	rad::DeviceVector<float> m_cuda;
	rad::DeviceVector<float> a_cuda;
	rad::DeviceVector<float> b_cuda;

	if (parameters.generate) {
		create_matrix(a_host, parameters.size);
		write_to_file(parameters.input, a_host);
	} else {
		read_from_file(parameters.input, a_host);
		read_from_file(parameters.gold, gold_final_vector);
		if (parameters.debug) {
			for (int i = 0; i < 10; i++) {
				gold_final_vector[i] = gold_final_vector[i] - 12;
				gold_final_vector[gold_final_vector.size() - i - 1] =
						gold_final_vector[i] - 12;
			}
		}
	}

	for (size_t iteration = 0; iteration < parameters.iterations; iteration++) {
		//setting the memory
		auto mem_set_time = rad::mysecond();
		m_cuda = m_host;
		a_cuda = a_host;
		b_cuda = b_host;
		mem_set_time = rad::mysecond() - mem_set_time;

		//begin timing
		auto kernel_time = rad::mysecond();

		log.start_iteration();
		// run kernels
		ForwardSub(m_cuda, a_cuda, b_cuda, parameters.size);
		//end timing
		log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		auto copy_time = rad::mysecond();
		// copy memory back to CPU
		a_cuda.to_vector(a_copy_host);
		b_cuda.to_vector(b_copy_host);

		copy_time = rad::mysecond() - copy_time;

		auto host_time = rad::mysecond();
		BackSub(final_vector, a_copy_host, b_copy_host, parameters.size);
		host_time = rad::mysecond() - host_time;

		size_t errors = 0;
		auto check_time = rad::mysecond();
		if (!parameters.generate)
			errors = check_and_log(final_vector, gold_final_vector, log,
					parameters.verbose);
		check_time = rad::mysecond() - check_time;

		if (parameters.verbose) {
			auto wasted_time = host_time + copy_time + mem_set_time
					+ check_time;
			auto overall_time = kernel_time + wasted_time;

			std::cout << "Overall time: " << overall_time << "s. ";
			std::cout << "Host part: " << host_time << "s. ";
			std::cout << "kernel time: " << kernel_time << "s. ";
			std::cout << "Time for copy: " << copy_time << "s. ";
			std::cout << "Check time: " << check_time << "s. ";
			std::cout << "Time for memset: " << mem_set_time << "s.\n";
			std::cout << "Iteration " << iteration << " - Errors " << errors
					<< " - Wasted time: " << wasted_time << "s. ("
					<< int(ceil((wasted_time / overall_time) * 100.0f)) << "%)"
					<< std::endl;
		}

	}

	if (parameters.generate) {
		write_to_file(parameters.gold, final_vector);
	}
}
