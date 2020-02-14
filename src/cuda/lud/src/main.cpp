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
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>

#include <vector>
#include <string>
#include <omp.h>

#include "generic_log.h" //from ../common/include
#include "device_vector.h"
#include "cuda_utils.h"

#include "Parameters.h"

#define BLOCK_SIZE 32

template<typename T>
void generateInputMatrix(std::vector<T>& array) {
	//TODO PLACE GENERATE INPUT FROM RODINIA

}

template<typename T>
bool read_from_file(std::string& path, std::vector<T>& array) {
	std::ifstream input(path, std::ios::binary);
	if (input.good()) {
		input.read(reinterpret_cast<char*>(array.data()),
				array.size() * sizeof(real_t));
		input.close();
		return false;
	}
	return true;
}

template<typename T>
bool write_to_file(std::string& path, std::vector<T>& array) {
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
	if (fabs(result) > 0.0000000001)
		return true;
	return false;
}

int main(int argc, char* argv[]) {
	Parameters parameters(argc, argv);

	std::string test_info = "size:" + std::to_string(parameters.size)
			+ " type:single-precision";
	std::string test_name = "cudaSLUD";
	//Log creation
	rad::Log log(testName, test_info);

	//================== Init test environment

	auto matrixSize = parameters.size * parameters.size;
	std::vector<float> INPUT(matrixSize);
	std::vector<float> OUTPUT(matrixSize);
	std::vector<float> GOLD(matrixSize);

	if (parameters.generate) {
		generateInputMatrix(INPUT);
	} else {
		auto read_input = read_from_file(parameters.input, INPUT);
		if (read_input == false) {
			throw_line("Error reading " + parameters.input);
		}
		auto read_gold = read_from_file(parameters.gold, GOLD);
		if (read_gold == false) {
			throw_line("Error reading " + parameters.gold);
		}
	}

	//====================================

	double total_kernel_time = 0;
	double min_kernel_time = UINT_MAX;
	double max_kernel_time = -999999;

	//================== Init DEVICE memory
	rad::DeviceVector<float> d_INPUT = INPUT;
	rad::DeviceVector<float> d_OUTPUT = INPUT;
	//====================================

	for (size_t loop2 = 0; loop2 < iterations; loop2++) { //================== Global test loop
		auto kernel_time = rad::mysecond();

		if (!parameters.generate)
			log.start_iteration();
		//================== Device computation, HMxM
		lud_cuda(d_INPUT, k);
		rad::checkCudaErrors(cudaDeviceSynchronize());
		rad::checkCudaErrors(cudaPeekAtLastError());
		//====================================

		if (!parameters.generate)
			log.end_iteration();
		kernel_time = rad::mysecond() - kernel_time;

		if (loop2 || !device_warmup) {
			total_kernel_time += kernel_time;
			min_kernel_time = min(min_kernel_time, kernel_time);
			max_kernel_time = max(max_kernel_time, kernel_time);
		}

		// Timer...
		auto cuda_copy_time = rad::mysecond();
		d_INPUT.to_vector(OUTPUT);
		cuda_copy_time = rad::mysecond() - cuda_copy_time;

		auto gold_check_time = rad::mysecond();
		if (generate) {
			write_to_file(parameters.gold, OUTPUT);
		} else {
			if (badass_memcmp(GOLD.data(), INPUT.data(), matrixSize)) {
				char error_detail[150];
				int host_errors = 0;

				printf("!");

#pragma omp parallel for
				for (i = 0; (i < k); i++) {
					for (j = 0; (j < k); j++) {
						if (INPUT[i + k * j] != GOLD[i + k * j])
#pragma omp critical
								{
							snprintf(error_detail, 150,
									"p: [%d, %d], r: %1.16e, e: %1.16e", i, j,
									(float) (INPUT[i + k * j]),
									(float) (GOLD[i + k * j]));
							if (verbose && (host_errors < 10))
								printf("%s\n", error_detail);

							log.log_error_detail(std::string(error_detail));
							host_errors++;
							//ea++;
							//fprintf(file, "\n p: [%d, %d], r: %1.16e, e: %1.16e, error: %d\n", i, j, A[i + k * j], GOLD[i + k * j], t_ea);

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
					d_OUTPUT = INPUT;
				}
				d_INPUT = d_OUTPUT;

			}
		}
		gold_check_time = gold_check_time - rad::mysecond();

		//====================================

		//================== Console hearthbeat
		std:cout << ".";
		//}
		//====================================
		if(parameters.verbose){
			std::cout << "Device kernel time for iteration " << loop2 << " - "<< kernel_time;
			std::cout << ". Gold check time " << gold_check_time;
			double outputpersec = (double) matrixSize / kernel_time;
			std::cout << "SIZE:%d OUTPUT/S:%f\n", k, outputpersec);
			printf("Iteration #%d time: %.3fs\n\n\n", loop2,
					mysecond() - global_time);

		}

	}

	double averageKernelTime = total_kernel_time
			/ (iterations - (device_warmup ? 1 : 0));
	printf("\n-- END --\n"
			"Total kernel time: %.3fs\n"
			"Iterations: %d\n"
			"Average kernel time: %.3fs (best: %.3fs ; worst: %.3fs)\n",
			total_kernel_time, iterations, averageKernelTime, min_kernel_time,
			max_kernel_time);


	return 0;
}
