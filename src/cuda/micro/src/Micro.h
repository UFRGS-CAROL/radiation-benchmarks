/*
 * MicroSpecial.h
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#ifndef MICROSPECIAL_H_
#define MICROSPECIAL_H_

#include <vector>
#include <random>
#include <iostream>
#include <sstream>      // std::stringstream
#include <iomanip> 	   // setprecision
#include <fstream> 	   // file operations

#include "Parameters.h"
#include "generic_log.h"
#include "device_vector.h"
#include "utils.h"

#include "omp.h"

template<typename ... Types> struct Input {
};

template<>
struct Input<double> {
//#define OPS_PER_THREAD_OPERATION 1
	double INPUT_A = 1.1945305291614955E+103; // 0x5555555555555555
	double INPUT_B = 3.7206620809969885E-103; // 0x2AAAAAAAAAAAAAAA
	double OUTPUT_R = 4.444444444444444; //0x4011C71C71C71C71
};

template<>
struct Input<float> {
//#define OPS_PER_THREAD_OPERATION 1
	float INPUT_A = 1.4660155E+13; // 0x55555555
	float INPUT_B = 3.0316488E-13; // 0x2AAAAAAA
	float OUTPUT_R = 4.444444; //0x408E38E3
};

template<>
struct Input<half> {
//#define OPS_PER_THREAD_OPERATION 2
	half INPUT_A = 1.066E+2; // 0x56AA
	half INPUT_B = 4.166E-2; // 0x2955
	half OUTPUT_R = 4.44; // 0x4471
};

template<typename real_t>
struct Micro {
	Parameters& parameters;
	std::shared_ptr<rad::Log>& log;
	const Input<real_t> input_limits;

	std::vector<real_t> input_host;
	rad::DeviceVector<real_t> input_device;

	std::vector<real_t> output_host;
	rad::DeviceVector<real_t> output_device;

	std::vector<real_t> gold;

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

		this->gold.resize(this->parameters.block_size, real_t(0.0));
		this->input_host.resize(this->parameters.block_size, real_t(0.0));

		if (this->parameters.generate == false) {
			this->read_from_file(this->parameters.input,
					this->input_host.data(), this->parameters.block_size);
			this->read_from_file(this->parameters.gold, this->gold.data(),
					this->parameters.block_size);
		} else {
			// First create an instance of an engine.
			std::random_device rnd_device;
			// Specify the engine and distribution.
			std::mt19937 mersenne_engine { rnd_device() }; // Generates random integers
			std::uniform_real_distribution<real_t> dist {
					-this->input_limits.OUTPUT_R, this->input_limits.OUTPUT_R };
			for (auto& i : this->input_host)
				i = dist(mersenne_engine) + real_t(0.001); //never zero
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
						+ std::string(typeid(real_t).name()))
	}

	size_t compare_output() {
		size_t errors = 0;

		if (this->parameters.generate == false) {
#pragma omp parallel for
			for (size_t i = 0; i < this->output_host.size(); i++) {
				real_t output = this->output_host[i];
				real_t gold_t = this->gold[i % this->parameters.block_size];
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
			this->write_to_file(this->parameters.input, this->input_host.data(),
					this->parameters.block_size, std::ios::out);

			this->write_to_file(this->parameters.gold, this->output_host.data(),
					this->parameters.block_size, std::ios::out);

		}
		return errors;
	}

	void reset_output_device() {
		this->output_device.clear();
	}

	bool read_from_file(std::string& path, real_t* array,
			uint32_t count) {
		std::ifstream input(path, std::ios::binary);
		if (input.good()) {
			input.read(reinterpret_cast<char*>(array), count * sizeof(real_t));
			input.close();
			return false;
		}
		return true;
	}

	template<typename openmode>
	bool write_to_file(std::string& path, real_t* array, uint32_t count,
			openmode& write_mode) {
		std::ofstream output(path, std::ios::binary | write_mode);
		if (output.good()) {
			output.write(reinterpret_cast<char*>(array),
					count * sizeof(real_t));
			output.close();

			return false;
		}
		return true;
	}

};

template<>
void Micro<float>::execute_micro();

#endif /* MicroSpecial_H_ */
