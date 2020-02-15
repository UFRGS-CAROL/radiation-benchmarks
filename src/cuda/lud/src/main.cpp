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
// CAROL-RADIATION radiation-benchmarks update made by fernandofernandesantos@gmail.com - 2020
#include <cuda.h>
#include <unistd.h>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <omp.h>

#include "generic_log.h" //from ../common/include
#include "device_vector.h"
#include "cuda_utils.h"

#include "Parameters.h"

#include "utils.h"

#define GET_RAND_FP (float(rand())/(float(RAND_MAX)+1.0f))

extern void lud_cuda(float *m, int matrix_dim);

template<typename T>
void generateInputMatrix(std::vector<T>& array, size_t size) {

	std::random_device rd; //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	//ORIGINAL:
	std::uniform_real_distribution<T> dis(1.0f, 2.0f);

	std::vector<T> L(array.size());
	std::vector<T> U(array.size());

	size_t i, j, k;

#pragma omp parallel for default(none) private(i,j) shared(L,U,size)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			if (i == j) {
				L[i * size + j] = 1.0;
				U[i * size + j] = GET_RAND_FP;
			} else if (i < j) {
				L[i * size + j] = 0;
				U[i * size + j] = GET_RAND_FP;
			} else { // i > j
				L[i * size + j] = GET_RAND_FP;
				U[i * size + j] = 0;
			}
		}
	}

#pragma omp parallel for default(none) private(i,j,k) shared(L,U,array,size)
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			T sum = 0;
			for (k = 0; k < size; k++)
				sum += L[i * size + k] * U[k * size + j];
			array[i * size + j] = sum;
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

bool badass_memcmp(float *gold, float *found, unsigned long n) {
	float result = 0.0;
	int i;
	unsigned long chunk = ceil(float(n) / float(omp_get_max_threads()));
	// printf("size %d max threads %d chunk %d\n", n, omp_get_max_threads(), chunk);
	double time = rad::mysecond();
#pragma omp parallel for default(shared) private(i) schedule(static,chunk) reduction(+:result)
	for (i = 0; i < n; i++)
		result = result + (gold[i] - found[i]);

	//  printf("comparing took %lf seconds, diff %lf\n", mysecond() - time, result);
	if (fabs(result) > 0.0000001)
		return true;
	return false;
}

int main(int argc, char* argv[]) {
	Parameters parameters(argc, argv);

	std::string test_info = "size:" + std::to_string(parameters.size)
			+ " type:single-precision";
	std::string test_name = "cudaSLUD";
	//Log creation
	rad::Log log(test_name, test_info);

	if (parameters.verbose) {
		std::cout << parameters << std::endl;
		std::cout << test_info << std::endl;
	}
	//================== Init test environment

	auto matrixSize = parameters.size * parameters.size;
	std::vector<float> INPUT(matrixSize);

	std::vector<float> GOLD(matrixSize);

	if (parameters.generate) {
		generateInputMatrix(INPUT, parameters.size);
		write_to_file(parameters.input, INPUT);
	} else {
		auto read_input = read_from_file(parameters.input, INPUT);
		if (read_input) {
			throw_line("Error reading " + parameters.input);
		}
		auto read_gold = read_from_file(parameters.gold, GOLD);
		if (read_gold) {
			throw_line("Error reading " + parameters.gold);
		}
	}
	std::vector<float> SAVE_INPUT = INPUT;
	//====================================

	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = -999999;

	//================== Init DEVICE memory
	rad::DeviceVector<float> d_INPUT = INPUT;
	rad::DeviceVector<float> d_OUTPUT = INPUT;
	//====================================

	for (size_t loop2 = 0; loop2 < parameters.iterations; loop2++) { //================== Global test loop
		auto kernel_time = rad::mysecond();

		if (!parameters.generate)
			log.start_iteration();
		//================== Device computation, HMxM
		lud_cuda(d_INPUT.data(), parameters.size);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		//====================================

		if (!parameters.generate)
			log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		if (loop2) {
			total_kernel_time += kernel_time;
			min_kernel_time = std::min(min_kernel_time, kernel_time);
			max_kernel_time = std::max(max_kernel_time, kernel_time);
		}

		// Timer...
		auto cuda_copy_time = rad::mysecond();
		d_INPUT.to_vector(INPUT);
		cuda_copy_time = rad::mysecond() - cuda_copy_time;

		auto gold_check_time = rad::mysecond();
		if (parameters.generate) {
			write_to_file(parameters.gold, INPUT);
		} else {
			if (badass_memcmp(GOLD.data(), INPUT.data(), matrixSize)) {
				char error_detail[150];
				int host_errors = 0;

				std::cout << "!";

#pragma omp parallel for
				for (size_t i = 0; i < parameters.size; i++) {
					for (size_t j = 0; j < parameters.size; j++) {
						auto g = GOLD[i + parameters.size * j];
						auto f = INPUT[i + parameters.size * j];
						if (g != f) {

							std::stringstream error_detail;
							error_detail << std::scientific
									<< std::setprecision(16);
							// "p: [%d, %d], r: %1.16e, e: %1.16e"
							error_detail << "p: [" << i << ", " << j << "], r: "
									<< f << ", e: " << g;

							if (parameters.verbose && (host_errors < 10))
								std::cout << error_detail.str() << std::endl;
#pragma omp critical
							{
								log.log_error_detail(error_detail.str());
								host_errors++;
								//ea++;
								//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + k * j], GOLD[i + k * j], t_ea);

							}
						}
					}
				}

				// printf("numErrors:%d", host_errors);

				log.update_errors();

				//================== Release device memory to ensure there is no corrupted data on the inputs of the next iteration

				if (host_errors != 0) {

					//================== Reload DEVICE memory
					d_INPUT.resize(0);
					d_OUTPUT.resize(0);
					d_OUTPUT = SAVE_INPUT;
				}
			}
			d_INPUT = d_OUTPUT;
		}
		gold_check_time = gold_check_time - rad::mysecond();

		//====================================

		//================== Console hearthbeat
		std::cout << ".";
		//}
		//====================================
		if (parameters.verbose) {
			std::cout << "Device kernel time for iteration " << loop2 << " - "
					<< kernel_time;
			std::cout << ".\nGold check time " << gold_check_time;
			double outputpersec = (double) matrixSize / kernel_time;
			std::cout << ".\nSIZE:" << parameters.size << " OUTPUT/S: "
					<< outputpersec;
			std::cout << "\nIteration " << loop2 << " time: "
					<< rad::mysecond() - total_kernel_time << std::endl;
			;

		}

	}

	auto averageKernelTime = total_kernel_time / parameters.iterations;
	std::cout << "\n-- END --\n";
	std::cout << "Total kernel time: " << total_kernel_time << std::endl;
	std::cout << "Iterations: " << parameters.iterations
			<< " Average kernel time: " << averageKernelTime << "s";
	std::cout << "(best: " << min_kernel_time << "s ; worst: "
			<< max_kernel_time << "s)" << std::endl;

	return 0;
}
