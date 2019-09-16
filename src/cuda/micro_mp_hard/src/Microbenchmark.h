/*
 * Microbenchmark.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef MICROBENCHMARK_H_
#define MICROBENCHMARK_H_

#include <tuple>
#include <sstream>
#include <omp.h>

#include "device_vector.h"
#include "Parameters.h"
#include "Log.h"
#include "none_kernels.h"

template<const uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct Microbenchmark {

	rad::DeviceVector<real_t> output_dev_1, output_dev_2, output_dev_3;
	std::vector<real_t> output_host_1, output_host_2, output_host_3;
	std::vector<real_t> gold_vector;

	Log log_;

	const Parameters& parameters_;

	Microbenchmark(const Parameters& parameters) :
			parameters_(parameters) {
		this->output_host_1.resize(this->parameters_.r_size);
		this->output_host_2.resize(this->parameters_.r_size);
		this->output_host_3.resize(this->parameters_.r_size);
		this->output_dev_1.resize(this->parameters_.r_size);
		this->output_dev_2.resize(this->parameters_.r_size);
		this->output_dev_3.resize(this->parameters_.r_size);

		this->gold_vector.resize(this->parameters_.r_size);
	}

//	virtual ~Microbenchmark() {
//	}

	double test() {
		auto kernel_time = rad::mysecond();

		//================== Device computation
		switch (parameters_.micro) {
		case ADD:
			microbenchmark_kernel_add<<<parameters_.grid_size,
					parameters_.block_size>>>(output_dev_1.data(),
					output_dev_2.data(), output_dev_3.data());
			break;
		case MUL:
			microbenchmark_kernel_mul<<<parameters_.grid_size,
					parameters_.block_size>>>(output_dev_1.data(),
					output_dev_2.data(), output_dev_3.data());
			break;
		case FMA:
			microbenchmark_kernel_fma<<<parameters_.grid_size,
					parameters_.block_size>>>(output_dev_1.data(),
					output_dev_2.data(), output_dev_3.data());
			break;
		}

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		return rad::mysecond() - kernel_time;
	}

	std::tuple<double, uint64, uint64> check_output_errors() {
		this->output_host_1 = this->output_dev_1.to_vector();
		this->output_host_2 = this->output_dev_2.to_vector();
		this->output_host_3 = this->output_dev_3.to_vector();

		if(this->parameters_.generate == true)
			return {0.0, 0, 0};

		auto cmp_time = rad::mysecond();
		uint64 host_errors = 0;
		uint64 memory_errors = 0;

#pragma omp parallel for shared(host_errors, memory_errors)
		for (uint32 i = 0; i < this->output_host_1.size(); i++) {
			register bool checkFlag = true;
			auto valGold = this->gold_vector[i];
			auto valOutput = this->output_host_1[i];

			auto valOutput0 = this->output_host_1[i];
			auto valOutput1 = this->output_host_2[i];
			auto valOutput2 = this->output_host_3[i];

			if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
#pragma omp critical
				{
//					char info_detail[150];
//					snprintf(info_detail, 150,
//							"m: [%d], r0: %1.20e, r1: %1.20e, r2: %1.20e", i,
//							(double) valOutput0, (double) valOutput1,
//							(double) valOutput2);
//					if (this->parameters_.verbose && (memory_errors < 10))
//						printf("%s\n", info_detail);
//					auto s = std::string(info_detail);
//					this->log_.log_info_detail(s);
//					memory_errors++;

					std::stringstream info_detail;
					info_detail.precision(20);
					info_detail << std::scientific;
					info_detail << "m: [" << i << "], r0: " << valOutput0;
					info_detail << ", r1: " << valOutput1;
					info_detail << ", r2: " << valOutput2;
					info_detail << ", e: " << valGold;

					if (this->parameters_.verbose && (host_errors < 10))
						std::cout << info_detail.str() << std::endl;
					this->log_.log_info_detail(info_detail.str());
					memory_errors++;
				}

				if ((valOutput0 != valOutput1) && (valOutput1 != valOutput2)
						&& (valOutput0 != valOutput2)) {
					// All 3 values diverge
					if (valOutput0 == valGold) {
						valOutput = valOutput0;
					} else if (valOutput1 == valGold) {
						valOutput = valOutput1;
					} else if (valOutput2 == valGold) {
						valOutput = valOutput2;
					} else {
						// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
						checkFlag = false;
#pragma omp critical
						{
							std::stringstream info_detail;
							info_detail.precision(20);
							info_detail << std::scientific;
							info_detail << "f: [" << i << "], r0: "
									<< valOutput0;
							info_detail << ", r1: " << valOutput1;
							info_detail << ", r2: " << valOutput2;
							info_detail << ", e: " << valGold;

							if (this->parameters_.verbose && (host_errors < 10))
								std::cout << info_detail.str() << std::endl;
							this->log_.log_info_detail(info_detail.str());
							memory_errors++;

//							char info_detail[150];
//							snprintf(info_detail, 150,
//									"f: [%d], r0: %1.20e, r1: %1.20e, r2: %1.20e, e: %1.20e",
//									i, (double) valOutput0, (double) valOutput1,
//									(double) valOutput2, (double) valGold);
//							if (this->parameters_.verbose && (host_errors < 10))
//								printf("%s\n", info_detail);
						}
					}
				} else if (valOutput1 == valOutput2) {
					// Only value 0 diverge
					valOutput = valOutput1;
				} else if (valOutput0 == valOutput2) {
					// Only value 1 diverge
					valOutput = valOutput0;
				} else if (valOutput0 == valOutput1) {
					// Only value 2 diverge
					valOutput = valOutput0;
				}
			}

			if (valGold != valOutput && checkFlag) {
#pragma omp critical
				{
					std::stringstream error_detail;
					error_detail.precision(20);
					error_detail << "p: [" << i << "], r: " << std::scientific
							<< valOutput << ", e: " << valGold;

					if (this->parameters_.verbose && (host_errors < 10))
						std::cout << error_detail.str() << std::endl;

					this->log_.log_error_detail(error_detail.str());
					host_errors++;
				}

			}
		}

		cmp_time -= rad::mysecond();

		this->log_.update_errors(host_errors);
		this->log_.update_infos(memory_errors);

		if (host_errors != 0) {
			std::cout << "#";
		}

		if (memory_errors != 0) {
			std::cout << "M";
		}

		return {cmp_time, host_errors, memory_errors};
	}

	uint64 check_which_one_is_right() {
		uint64 memory_errors = 0;

#pragma omp parallel for shared(memory_errors)
		for (uint32 i = 0; i < this->gold_vector.size(); i++) {
			auto valOutput0 = this->output_host_1[i];
			auto valOutput1 = this->output_host_2[i];
			auto valOutput2 = this->output_host_3[i];

			if ((valOutput0 != valOutput1) || (valOutput0 != valOutput2)) {
#pragma omp critical
				{
					memory_errors++;
				}

				if (valOutput1 == valOutput2) {
					// Only value 0 diverge
					this->gold_vector[i] = valOutput1;
				} else if (valOutput0 == valOutput2) {
					// Only value 1 diverge
					this->gold_vector[i] = valOutput0;
				} else if (valOutput0 == valOutput1) {
					// Only value 2 diverge
					this->gold_vector[i] = valOutput0;
				}
			}
		}

		return memory_errors;
	}

	void write_gold() {
		auto memory_errors = this->check_which_one_is_right();
		if (memory_errors != 0) {
			std::string err = "GOLDEN GENERATOR FAILED "
					+ std::to_string(memory_errors);
			fatalerror(err.c_str());
		}

		this->write_to_file(this->parameters_.gold_file, this->gold_vector);
	}

	void load_gold() {
		this->load_file_data(this->parameters_.gold_file, this->gold_vector);
	}

	void load_file_data(std::string path, std::vector<real_t>& array) {
		std::ifstream input(path, std::ios::binary);
		if (input.good()) {
			input.read(reinterpret_cast<char*>(array.data()),
					array.size() * sizeof(real_t));
		}
		input.close();
	}

	void write_to_file(std::string path, std::vector<real_t>& array) {
		std::ofstream output(path, std::ios::binary);
		if (output.good()) {
			output.write(reinterpret_cast<char*>(array.data()),
					array.size() * sizeof(real_t));
		}
		output.close();
	}

};

#endif /* MICROBENCHMARK_H_ */
