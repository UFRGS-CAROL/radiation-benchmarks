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

	std::vector<micro_type_t> output_host;
	rad::DeviceVector<micro_type_t> output_device;

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

		auto start_gen = rad::mysecond();
		//Set the output size
		this->output_device.resize(this->parameters.array_size);
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
		std::vector<micro_type_t> output_to_save(this->output_host.begin(),
				this->output_host.begin() + this->block_size);

		write_to_file(this->parameters.gold, output_to_save);
	}

	virtual ~Micro() = default;

	void copy_back_output() {
		this->output_device.to_vector(this->output_host);
	}

	virtual void execute_micro() {
		throw_line(
				"Method execute_micro not created for type == "
						+ std::string(typeid(micro_type_t).name()))
	}

	size_t compare_output() {
		size_t errors = 0;
		auto& block_size = this->parameters.block_size;
#pragma omp parallel for
		for (size_t i = 0; i < this->output_host.size(); i++) {
			micro_type_t output = this->output_host[i];
			micro_type_t gold_t = this->gold[i % block_size];
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

					this->log->log_error_detail(error_detail.str());
				}
			}
		}
		return errors;
	}

	void reset_output_device() {
		this->output_device.clear();
	}

};

#endif /* MicroSpecial_H_ */
