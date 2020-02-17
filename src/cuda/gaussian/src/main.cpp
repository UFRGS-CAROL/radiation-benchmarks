#include <vector>
#include <iostream>
#include <omp.h>
#include <fstream>

#include "cuda_utils.h"
#include "generic_log.h"
#include "Parameters.h"
#include "utils.h"

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
void check_and_log(std::vector<real_t>& final_vector,
		std::vector<real_t>& gold_final_vector, rad::Log& log) {


#pragma omp parallel for default(shared)
	for(size_t i = 0; i < gold_final_vector.size(); i++){

	}
}

int main(int argc, char *argv[]) {

	Parameters parameters(argc, argv);
	std::string test_name = "cudaGaussian";
	std::string test_info = "size: " + std::to_string(parameters.size);
	test_info += "maxblocksize: " + std::to_string(MAXBLOCKSIZE);
	test_info += " blocksizexy: " + std::to_string(BLOCK_SIZE_XY);

	rad::Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << test_info << std::endl;
	}

	size_t matrix_size = parameters.size * parameters.size;
	std::vector<float> a_host(matrix_size);
	std::vector<float> m_host(matrix_size, 0.0f);

	std::vector<float> b_host(parameters.size, 1.0f);
	std::vector<float> final_vector(parameters.size);
	std::vector<float> gold_final_vector(parameters.size);

	std::cout << "WG size of kernel 1 = " << MAXBLOCKSIZE
			<< ", WG size of kernel 2= " << BLOCK_SIZE_XY << " X "
			<< BLOCK_SIZE_XY << std::endl;

	if (parameters.generate) {
		create_matrix(a_host, parameters.size);
		write_to_file(parameters.input, a_host);
	} else {
		read_from_file(parameters.input, a_host);
		read_from_file(parameters.gold, gold_final_vector);
	}
	rad::DeviceVector<float> m_cuda = m_host;
	rad::DeviceVector<float> a_cuda = a_host;
	rad::DeviceVector<float> b_cuda = b_host;

	//begin timing
	auto kernel_time = rad::mysecond();

	std::cout << "Starting forward Sub\n";
	log.start_iteration();
	// run kernels
	ForwardSub(m_cuda, a_cuda, b_cuda, parameters.size);
	//end timing
	log.end_iteration();
	kernel_time = rad::mysecond() - kernel_time;

	auto copy_time = rad::mysecond();
	// copy memory back to CPU
	m_cuda.to_vector(m_host);
	a_cuda.to_vector(a_host);
	b_cuda.to_vector(b_host);

	copy_time = rad::mysecond() - copy_time;

	std::cout << "End forward Sub\n";

	std::cout << "Starting Back Sub\n";
	auto host_part = rad::mysecond();

	BackSub(final_vector, a_host, b_host, parameters.size);

	host_part = rad::mysecond() - host_part;
	if (parameters.verbose) {
		printf("The final solution is: \n");
		for (auto i : final_vector)
			std::cout << i << " ";
		std::cout << std::endl;
	}

	auto overall_time = kernel_time + host_part + copy_time;

	std::cout << "Time total " << overall_time << "s\n";
	std::cout << "Host part " << host_part << "s\n";
	std::cout << "Time for CUDA kernels: " << kernel_time << "s\n";
	std::cout << "Time for copy " << copy_time << "s\n";

	if(parameters.generate){
		write_to_file(parameters.gold, final_vector);
	}
}
