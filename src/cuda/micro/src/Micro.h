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
#include <fstream> 	   // file operations

#include "Parameters.h"
#include "generic_log.h"
#include "device_vector.h"
#include "utils.h"
#include "omp.h"

template<typename micro_type_t>
struct Micro {
	Parameters& parameters;
	std::shared_ptr<rad::Log>& log;
	const Input<micro_type_t> input_limits;

	std::vector<micro_type_t> input_host;
	rad::DeviceVector<micro_type_t> input_device;

	std::vector<micro_type_t> output_host;
	rad::DeviceVector<micro_type_t> output_device;

	std::vector<micro_type_t> gold;

	size_t grid_size;
	size_t block_size;

	Micro(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			parameters(parameters), log(log) {
		auto start_gen = rad::mysecond();
		//Set the output size
		this->output_device.resize(this->parameters.array_size);
		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Output device allocation time: "
					<< end_gen - start_gen << std::endl;
		}

		this->gold.resize(this->parameters.block_size, micro_type_t(0.0));
		this->input_host.resize(this->parameters.block_size, micro_type_t(0.0));

		if (this->parameters.generate == false) {
			this->read_from_file(this->parameters.input, this->input_host);
			this->read_from_file(this->parameters.gold, this->gold);
		} else {

			if (file_exists(this->parameters.input)) {
				if (this->parameters.verbose) {
					std::cout << this->parameters.input
							<< " file already exists, reading\n";
				}
				this->read_from_file(this->parameters.input, this->input_host);
			} else {
				if (this->parameters.verbose) {
					std::cout << "Generating a new input file\n";
				}

				this->input_host.resize(this->parameters.block_size);
				generate_new_array(this->input_host);
				this->write_to_file(this->parameters.input, this->input_host,
						std::ios::out);
			}
		}

		this->input_device = this->input_host;

	}

	virtual ~Micro() = default;

	void copy_back_output() {
		this->output_device.to_vector(this->output_host);
	}

	void execute_micro() {
		throw_line(
				"Method execute_micro not created for type == "
						+ std::string(typeid(micro_type_t).name()))
	}

	size_t compare_output() {
		size_t errors = 0;

		if (this->parameters.generate == false) {
#pragma omp parallel for
			for (size_t i = 0; i < this->output_host.size(); i++) {
				micro_type_t output = this->output_host[i];
				micro_type_t gold_t =
						this->gold[i % this->parameters.block_size];
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
		} else {
			//save only the first thread result
			//This will save only the first BLOCK_SIZE of results
			//which must be equals to the rest of the array
			this->write_to_file(this->parameters.gold, this->output_host, std::ios::out);

		}
		return errors;
	}

	void reset_output_device() {
		this->output_device.clear();
	}

	bool read_from_file(std::string& path, std::vector<micro_type_t>& array) {
		size_t count = array.size();
		std::ifstream input(path, std::ios::binary);
		if (input.good()) {
			input.read(reinterpret_cast<char*>(array.data()),
					count * sizeof(micro_type_t));
			input.close();
			return false;
		}
		return true;
	}

	template<typename openmode>
	bool write_to_file(std::string& path, std::vector<micro_type_t>& array,
			openmode& write_mode) {
		size_t count = array.size();
		std::ofstream output(path, std::ios::binary | write_mode);
		if (output.good()) {
			output.write(reinterpret_cast<char*>(array.data()),
					count * sizeof(micro_type_t));
			output.close();

			return false;
		}
		return true;
	}

};

#endif /* MicroSpecial_H_ */
