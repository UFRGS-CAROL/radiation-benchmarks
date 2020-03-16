/*
 * MicroSpecial.h
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#ifndef MICROSPECIAL_H_
#define MICROSPECIAL_H_

#include <iostream>
#include <sstream>      // std::stringstream
#include <iomanip> 	   // setprecision

#include "Parameters.h"
#include "generic_log.h"
#include "device_vector.h"
#include "utils.h"
#include "omp.h"

template<typename micro_type_t>
struct Micro {
	Parameters& parameters;
	std::shared_ptr<rad::Log>& log;
	bool is_ecc_on;

	std::vector<micro_type_t> output_host_1;
	std::vector<micro_type_t> output_host_2;
	std::vector<micro_type_t> output_host_3;

	rad::DeviceVector<micro_type_t> output_device_1;
	rad::DeviceVector<micro_type_t> output_device_2;
	rad::DeviceVector<micro_type_t> output_device_3;

	std::vector<micro_type_t> gold;

	size_t grid_size;
	size_t block_size;

	Micro();

	Micro(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			parameters(parameters), log(log) {
		//all benchmarks will use MAX_THREAD_BLOCK size
		//Max thread block size set
		this->block_size = MAX_THREAD_BLOCK;
		//multiplies the grid size by the maximum number of warps per SM
		this->grid_size = this->parameters.sm_count * WARP_PER_SM;
		this->parameters.array_size = this->grid_size * this->block_size;

		this->is_ecc_on = (this->log.get_log_file_name().find("ECC_ON")
				!= std::string::npos);
		auto start_gen = rad::mysecond();
		//Set the output size
		this->output_device_1.resize(this->parameters.array_size);
		this->output_device_2.resize(this->parameters.array_size);
		this->output_device_3.resize(this->parameters.array_size);

		this->gold.resize(this->parameters.block_size);

		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Output/input device allocation time: "
					<< end_gen - start_gen << std::endl;
		}
	}

	void get_setup_input() {
		read_from_file(this->parameters.gold, this->gold);
	}

	void save_output() {
		//save only the first thread result
		//This will save only the first BLOCK_SIZE of results
		//which must be equals to the rest of the array
		std::vector<micro_type_t> output_to_save(this->output_host_1.begin(),
				this->output_host_1.begin() + this->block_size);

		write_to_file(this->parameters.gold, output_to_save);
	}

	virtual ~Micro() = default;

	void copy_back_output() {
		this->output_device_1.to_vector(this->output_host_1);
		if (this->is_ecc_on == false) {
			this->output_device_2.to_vector(this->output_host_2);
			this->output_device_3.to_vector(this->output_host_3);
		}
	}

	virtual void execute_micro() {
		throw_line(
				"Method execute_micro not created for type == "
						+ std::string(typeid(micro_type_t).name()))
	}

	size_t compare_output() {
		size_t errors = 0;
		size_t memory_errors = 0;
		auto& block_size = this->parameters.block_size;
#pragma omp parallel for default(shared)
		for (size_t i = 0; i < this->output_host_1.size(); i++) {
			micro_type_t output = this->output_host_1[i];
			micro_type_t gold_t = this->gold[i % block_size];

			if (this->is_ecc_on == false) {
				output = this->check_mem_errors(gold_t, memory_errors);
			}
			cmp_and_log_diff(output, gold_t, i, errors);
		}


		if (memory_errors != 0) {
			if (this->parameters.verbose)
				std::cout << "M";
			std::string info_detail = "memory_positions_corrupted:"
					+ std::to_string(memory_errors);
			this->log.log_info_detail(info_detail);
			this->log.update_infos();

		}

		return errors;
	}

	void reset_output_device() {
		this->output_device_1.clear();
	}

protected:
	void inline cmp_and_log_diff(micro_type_t output, micro_type_t gold_t,
			size_t i, size_t& errors) {
		if (output != gold_t) {
			std::stringstream error_detail;
			//20 is from old micro-benchmarks precision
			error_detail << " p: [" << i << "],";
			error_detail << std::scientific << std::setprecision(20);
			error_detail << " e: " << gold_t << ", r: " << output;
			if (this->parameters.verbose && i < 10) {
				std::cout << error_detail.str() << std::endl;
			}

#pragma omp critical
			{
				errors++;
				this->log.log_error_detail(error_detail.str());
			}
		}
	}

	micro_type_t check_mem_errors(micro_type_t val_gold,
			size_t& memory_errors) {

		auto val_output_1 = this->output_host_1[i];
		auto val_output_2 = this->output_host_2[i];
		auto val_output_3 = this->output_host_3[i];
		auto val_output = val_output_1;

		if ((val_output_1 != val_output_2) || (val_output_1 != val_output_3)) {
#pragma omp critical
			{
				//					std::stringstream info_detail;
				//					info_detail.precision(PRECISION_PLACES);
				//					info_detail << std::scientific;
				//					info_detail << "m: [" << i << "], r0: " << val_output_1;
				//					info_detail << ", r1: " << val_output_2;
				//					info_detail << ", r2: " << val_output_3;
				//					info_detail << ", e: " << val_gold;
				//
				//					if (this->parameters_.verbose && (host_errors < 10))
				//						std::cout << info_detail.str() << std::endl;
				//					this->log_.log_info_detail(info_detail.str());
				memory_errors++;
			}

			if ((val_output_1 != val_output_2) && (val_output_2 != val_output_3)
					&& (val_output_1 != val_output_3)) {
				// All 3 values diverge
				if (val_output_1 == val_gold) {
					val_output = val_output_1;
				} else if (val_output_2 == val_gold) {
					val_output = val_output_2;
				} else if (val_output_3 == val_gold) {
					val_output = val_output_3;
				} else {
					// NO VALUE MATCHES THE GOLD AND ALL 3 DIVERGE!
					check_flag = false;
#pragma omp critical
					{
						//							std::stringstream info_detail;
						//							info_detail.precision(PRECISION_PLACES);
						//							info_detail << std::scientific;
						//							info_detail << "f: [" << i << "], r0: "
						//									<< val_output_1;
						//							info_detail << ", r1: " << val_output_2;
						//							info_detail << ", r2: " << val_output_3;
						//							info_detail << ", e: " << val_gold;
						//
						//							if (this->parameters_.verbose && (host_errors < 10))
						//								std::cout << info_detail.str() << std::endl;
						//							this->log_.log_info_detail(info_detail.str());
						memory_errors++;
					}
				}
			} else if (val_output_2 == val_output_3) {
				// Only value 0 diverge
				val_output = val_output_2;
			} else if (val_output_1 == val_output_3) {
				// Only value 1 diverge
				val_output = val_output_1;
			} else if (val_output_1 == val_output_2) {
				// Only value 2 diverge
				val_output = val_output_1;
			}
		}

		return val_output;
	}
};

#endif /* MicroSpecial_H_ */
