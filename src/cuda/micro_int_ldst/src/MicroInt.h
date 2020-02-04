/*
 * MicroInt.h
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#ifndef MICROINT_H_
#define MICROINT_H_

#include <vector>

#include <random>
#include <iostream>

#include "Parameters.h"
#include "device_vector.h"
#include "utils.h"

__device__ static unsigned long long errors;

template<typename int_t>
struct MicroInt {
	Parameters& parameters;

	std::vector<int_t> input_host;
	std::vector<int_t> output_host;

	rad::DeviceVector<int_t> input_device;
	rad::DeviceVector<int_t> output_device;

	uint32_t grid_size, block_size, operation_num;
	size_t array_size;

	MicroInt(Parameters& parameters) :
			parameters(parameters) {
		//Setting input and output host and device
		this->block_size = MAX_THREAD_BLOCK;

		if (this->parameters.micro == LDST) {
			this->grid_size = this->parameters.sm_count;
			//Input and output arrays
			this->array_size = this->parameters.global_gpu_memory_bytes
					/ (sizeof(int_t) * 3); //two arrays in the memory + safe space

			this->operation_num = this->array_size
					/ (this->grid_size * this->block_size);
		} else {
			this->grid_size = WARP_PER_SM * this->parameters.sm_count;
			this->array_size = WARP_PER_SM * this->parameters.sm_count
					* MAX_THREAD_BLOCK;
			this->operation_num = OPS;
		}

		auto start_gen = rad::mysecond();
		this->generate_input();
		auto end_gen = rad::mysecond();

		//Set the size of
//		this->output_host.resize(this->array_size);
		this->output_device.resize(this->array_size);
		this->input_device = this->input_host;

		if (this->parameters.verbose) {
			std::cout << "Input generation time: " << end_gen - start_gen
					<< std::endl;
		}
	}

	void generate_input() {
		// First create an instance of an engine.
		std::random_device rnd_device;
		// Specify the engine and distribution.
		std::mt19937 mersenne_engine { rnd_device() }; // Generates random integers
		std::uniform_int_distribution<int_t> dist { 1, RANGE_INT_VAL };
		std::vector<int_t> temp_input(MAX_THREAD_BLOCK);
		for (auto& i : temp_input)
			i = dist(mersenne_engine);

		if (this->parameters.micro == LDST) {
			this->input_host.resize(this->array_size);
			uint32_t slice = this->array_size / temp_input.size();

			for (uint32_t i = 0; i < this->array_size; i += slice) {
					std::copy(temp_input.begin(), temp_input.end(),
							this->input_host.begin() + i);
			}

		} else {
			this->input_host = temp_input;
		}
//		std::cout << "INPUT SIZE <<<<< " << this->input_host.size()
//				<< std::endl;
//		std::cout << this->grid_size << " " << this->block_size << std::endl;
	}

	virtual ~MicroInt() = default;

	size_t compare_output() {
		return this->compare_on_gpu();
	}
	void copy_back_output() {
		//this->output_host = this->output_device.to_vector();
	}

	void execute_micro() {

	}

	size_t compare_on_gpu(){
		return 0;
	}
};

template<>
void MicroInt<int32_t>::execute_micro();
template<>
size_t MicroInt<int32_t>::compare_on_gpu();

#endif /* MICROINT_H_ */
