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
#include "generic_log.h"
#include "device_vector.h"
#include "utils.h"

#include "omp.h"

__device__ static unsigned long long errors;

template<typename int_t>
struct MicroInt {
	Parameters& parameters;
	std::shared_ptr<rad::Log>& log;

	size_t grid_size;
	size_t block_size;
	size_t array_size;

	std::vector<int_t> gold_host;
	std::vector<int_t> gold_branch_kernel;
	std::vector<int_t> input_host;
	std::vector<int_t> output_host;

	rad::DeviceVector<int_t> input_device;
	rad::DeviceVector<int_t> output_device;

	MicroInt(Parameters& parameters, std::shared_ptr<rad::Log>& log) :
			parameters(parameters), log(log) {
		//both benchmarks will use MAX_THREAD_BLOCK size
		this->block_size = MAX_THREAD_BLOCK;

		if (this->parameters.micro == LDST) {
			//Array size is the amount of memory that will be evaluated
			this->array_size = this->parameters.memory_size_to_use
					/ sizeof(int_t);
			this->parameters.operation_num = MEM_OPERATION_NUM;
			this->grid_size = this->array_size
					/ (this->parameters.operation_num * this->block_size);
		} else {
			//multiplies the grid size by the maximum number of warps per SM
			this->grid_size = this->parameters.sm_count * WARP_PER_SM;
			this->array_size = this->grid_size * this->block_size;
		}

		auto start_gen = rad::mysecond();
		this->generate_input();
		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Input generation time: " << end_gen - start_gen
					<< std::endl;
		}

		start_gen = rad::mysecond();
		//Set the output size
		this->output_device.resize(this->array_size);
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
		std::uniform_int_distribution<int_t> dist { RANGE_INT_MIN, RANGE_INT_MAX };

		//generates both golds
		for (auto i = 0; i < MAX_THREAD_BLOCK; i++) {
			this->gold_branch_kernel.push_back(i);
			this->gold_host.push_back(dist(mersenne_engine));
		}

		if (this->parameters.micro == LDST) {
			this->input_host.resize(this->array_size);
			for (auto begin = this->input_host.begin(), end =
					this->input_host.end(); begin < end;
					begin += this->gold_host.size()) {
				std::copy(this->gold_host.begin(), this->gold_host.end(),
						begin);
			}
		} else {
			this->input_host = this->gold_host;
		}

		if (parameters.debug) {
			this->input_host[rand() % this->input_host.size()] = 0;
		}

		this->input_device = this->input_host;
	}

	virtual ~MicroInt() = default;

	void copy_back_output() {
		this->output_device.to_vector(this->output_host);
	}

	void execute_micro() {
		throw_line(
				"Method execute_micro not created for type == "
						+ std::string(typeid(int_t).name()))
	}

	size_t compare_on_gpu() {
		throw_line(
				"Method compare_on_gpu() not created for type == "
						+ std::string(typeid(int_t).name()))
		return 0;
	}

	size_t compare_output() {
		auto gold_size = this->gold_host.size();
		auto slices = this->array_size / gold_size;
		std::vector<size_t> error_vector(slices, 0);
		size_t slice;

		auto gold_ptr = this->gold_host.data();
		if (this->parameters.micro == BRANCH)
			gold_ptr = this->gold_branch_kernel.data();

#pragma omp parallel for shared(error_vector, slice)
		for (slice = 0; slice < slices; slice++) {
			auto i_ptr = slice * gold_size;
			int_t* output_ptr = this->output_host.data() + i_ptr;

			for (size_t i = 0; i < this->gold_host.size(); i++) {
				int_t output = output_ptr[i];
				int_t golden = gold_ptr[i];
				if (output != golden) {
					std::string error_detail;
					error_detail = "array_position: " + std::to_string(i_ptr);
					error_detail += " gold_position: " + std::to_string(i);
					error_detail += " e: " + std::to_string(golden);
					error_detail += " r: " + std::to_string(output);

					if (this->parameters.verbose && error_vector[i] < 5) {
						std::cout << error_detail << std::endl;
					}
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
void MicroInt<int32_t>::execute_micro();

//template<>
//size_t MicroInt<int32_t>::compare_on_gpu();

#endif /* MICROINT_H_ */
