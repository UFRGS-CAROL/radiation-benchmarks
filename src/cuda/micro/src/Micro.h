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
	const Input<real_t> input_kernel;
	real_t gold;

	std::vector<real_t> output_host;
	rad::DeviceVector<real_t> output_device;

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

		switch (this->parameters.micro) {
		case ADD:
		case MUL:
		case FMA:
			this->gold = this->input_kernel.OUTPUT_R;
			break;
		case PYTHAGOREAN:
		case EULER:
			if (this->parameters.generate == false) {
				this->read_from_file(this->parameters.generate_output,
						&(this->gold), 1);
			}
			break;
		}

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
				if (output != this->gold) {
					std::stringstream error_detail;
					//20 is from old micro-benchmarks precision
					error_detail << " p: [" << i << "],";
					error_detail << std::scientific << std::setprecision(20);
					error_detail << " e: " << this->gold << ", r: " << output;

					if (this->parameters.verbose && i < 10) {
						std::cout << error_detail.str() << std::endl;
					}
					errors++;
#pragma omp critical
					{
						this->log->log_error_detail(error_detail.str());
					}
				}
			}
		} else {
			//save only the first thread result
			this->gold = this->output_host[0];
			if (this->parameters.micro == EULER
					|| this->parameters.micro == PYTHAGOREAN) {
				this->write_to_file(this->parameters.generate_output,
						&(this->gold), 1);
			}
		}
		return errors;
	}

	void reset_output_device() {
		this->output_device.clear();
	}

	static bool read_from_file(std::string& path, real_t* array,
			uint32_t count) {
		std::ifstream input(path, std::ios::binary);
		if (input.good()) {
			input.read(reinterpret_cast<char*>(array), count * sizeof(real_t));
			input.close();
			return false;
		}
		return true;
	}

	static bool write_to_file(std::string& path, real_t* array,
			uint32_t count) {
		std::ofstream output(path, std::ios::binary);
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
