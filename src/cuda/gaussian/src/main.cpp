#include <vector>
#include <iostream>

#include "cuda_utils.h"
#include "Parameters.h"

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
/**
 * Benchmark from rodinia benchmark suite,
 *  updated for radiation benchmarks by Fernando (fernandofernandesantos@gmail.com) 2020
 */

#include "utils.h"

template<typename real_t>
void create_matrix(std::vector<real_t>& m, size_t size) {
	int i, j;
	real_t lamda = -0.01;
	std::vector<real_t> coe(2 * size - 1);
	real_t coe_i = 0.0;

	for (i = 0; i < size; i++) {
		coe_i = 10 * exp(lamda * i);
		j = size - 1 + i;
		coe[j] = coe_i;
		j = size - 1 - i;
		coe[j] = coe_i;
	}

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			m[i * size + j] = coe[size - 1 - i + j];
		}
	}

}

int main(int argc, char *argv[]) {

	Parameters parameters(argc, argv);

	size_t matrix_size = parameters.size * parameters.size;
	std::vector<float> a(matrix_size);
	std::vector<float> b(matrix_size, 1.0f);
	std::vector<float> m(matrix_size, 0.0f);
	std::vector<float> finalVec(parameters.size);

	rad::DeviceVector<float> m_cuda = m;
	rad::DeviceVector<float> a_cuda = a;
	rad::DeviceVector<float> b_cuda = b;

	std::cout << "WG size of kernel 1 = " << MAXBLOCKSIZE
			<< ", WG size of kernel 2= " << BLOCK_SIZE_XY << " X "
			<< BLOCK_SIZE_XY << std::endl;

	create_matrix(a, parameters.size);
	//begin timing
	auto time_start = rad::mysecond();

	std::cout << "Starting forward Sub\n";
	// run kernels
	ForwardSub(m_cuda, a_cuda, b_cuda, parameters.size);
	//end timing

	auto kernel_time = rad::mysecond() - time_start;
	// copy memory back to CPU
	m_cuda.to_vector(m);
	a_cuda.to_vector(a);
	b_cuda.to_vector(b);

	std::cout << "End forward Sub\n";

	std::cout << "Starting Back Sub\n";
	BackSub(finalVec, a, b, parameters.size);
	if (parameters.verbose) {
		printf("The final solution is: \n");
		for (auto i : finalVec)
			std::cout << i << " ";
		std::cout << std::endl;
	}

	auto time_end = rad::mysecond();

	std::cout << "Time total (including memory transfers) "
			<< (time_end - time_start) << "s\n";

	std::cout << "Time for CUDA kernels: " << kernel_time << "s\n";
}
