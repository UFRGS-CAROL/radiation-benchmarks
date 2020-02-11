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

#include "Parameters.h"
#include "generic_log.h"
#include "device_vector.h"
#include "utils.h"

#include "omp.h"

#if defined(test_precision_double)

#define OPS_PER_THREAD_OPERATION 1
#define INPUT_A 1.1945305291614955E+103 // 0x5555555555555555
#define INPUT_B 3.7206620809969885E-103 // 0x2AAAAAAAAAAAAAAA
#define OUTPUT_R 4.444444444444444 //0x4011C71C71C71C71
const char test_precision_description[] = "double";
typedef double tested_type;
typedef double tested_type_host;

#elif defined(test_precision_single)

#define OPS_PER_THREAD_OPERATION 1
#define INPUT_A 1.4660155E+13 // 0x55555555
#define INPUT_B 3.0316488E-13 // 0x2AAAAAAA
#define OUTPUT_R 4.444444 //0x408E38E3
const char test_precision_description[] = "single";
typedef float tested_type;
typedef float tested_type_host;

#elif defined(test_precision_half)

#define OPS_PER_THREAD_OPERATION 2
#define INPUT_A 1.066E+2 // 0x56AA
#define INPUT_B 4.166E-2 // 0x2955
#define OUTPUT_R 4.44 // 0x4471
const char test_precision_description[] = "half";
typedef half tested_type;
typedef half_float::half tested_type_host;

#endif

template<typename real_t>
struct Micro {
	Parameters& parameters;
	std::shared_ptr<rad::Log>& log;

	std::vector<real_t> gold_host;
	std::vector<real_t> output_host;
	rad::DeviceVector<real_t> output_device;

	Micro(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			parameters(parameters), log(log) {
		auto start_gen = rad::mysecond();
		this->generate_input();
		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Input generation time: " << end_gen - start_gen
					<< std::endl;
		}

		start_gen = rad::mysecond();
		//Set the output size
		this->output_device.resize(this->parameters.array_size);
		end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Output device allocation time: "
					<< end_gen - start_gen << std::endl;
		}

	}

	void generate_input() {
		// First create an instance of an engine.
		std::random_device rnd_device;
		// Specify the engine and distribution.
		std::mt19937 mersenne_engine { rnd_device() }; // Generates random integers
		std::uniform_int_distribution<real_t> dist { 1, RANGE_INT_VAL };
		this->gold_host.resize(parameters.block_size, 0);

		for (auto& i : this->gold_host)
			i = dist(mersenne_engine);
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
		auto gold_size = this->gold_host.size();
		auto slices = this->parameters.array_size / gold_size;
		std::vector < size_t > error_vector(slices, 0);
		size_t slice;
		std::cout << "NUM OF SLICES " << slices << std::endl;

#pragma omp parallel for shared(error_vector, slice)
		for (slice = 0; slice < slices; slice++) {
			auto i_ptr = slice * gold_size;
			real_t* output_ptr = this->output_host.data() + i_ptr;

			for (size_t i = 0; i < this->gold_host.size(); i++) {
				real_t output = output_ptr[i];
				real_t golden = this->gold_host[i];
				if (output != golden) {

//					char error_detail[150];
//					snprintf(error_detail, 150,
//							"p: [%d], r: %1.20e, e: %1.20e",
//							i, (double)valOutput, (double)valGold);
//					if (verbose && (host_errors < 10))
//						printf("%s\n", error_detail);

					std::string error_detail;
					error_detail = "array_position: " + std::to_string(i_ptr);
					error_detail += " gold_position: " + std::to_string(i);
					error_detail += " e: " + std::to_string(golden);
					error_detail += " r: " + std::to_string(output);
					std::cout << error_detail << std::endl;
					error_vector[i]++;
#pragma omp critical
					{
						this->log->log_error_detail(error_detail);
					}
				}
			}
		}

		return std::accumulate(error_vector.begin(), error_vector.end(), 0);
	}

	void reset_output_device() {
		this->output_device.clear();
	}

};

template<>
void Micro<float>::execute_micro();

//template<>
//size_t MicroSpecial<int32_t>::compare_on_gpu();

#endif /* MicroSpecial_H_ */
