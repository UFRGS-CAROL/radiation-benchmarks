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

#include <thread>         // std::thread
#include <mutex>          // std::mutex, std::unique_lock, std::defer_lock

#include "Parameters.h"
#include "Log.h"
#include "device_vector.h"
#include "utils.h"

#include <omp.h>

__device__ static unsigned long long errors;

template<typename int_t>
struct MicroInt {
	Parameters& parameters;
	Log& log;

	//mutex for host cmp
	std::mutex thread_mutex;

	std::vector<int_t> input_gold_host;
	std::vector<int_t> output_host;

	rad::DeviceVector<int_t> input_device;
	rad::DeviceVector<int_t> output_device;

	uint32_t grid_size, block_size, operation_num;
	size_t array_size;

	MicroInt(Parameters& parameters, Log& log) :
			parameters(parameters), log(log) {
		//both benchmarks will use MAX_THREAD_BLOCK size
		this->block_size = MAX_THREAD_BLOCK;

		if (this->parameters.micro == LDST) {
			//Array size is the amount of memory that will be evaluated
			this->array_size = GPU_DDR_TEST_SIZE / sizeof(int_t);
			this->operation_num = MEM_OPERATION_NUM;
			this->grid_size = this->array_size
					/ (this->operation_num * this->block_size);
		} else {
			//multiplies the grid size by the maximum number of warps per SM
			this->grid_size = this->parameters.sm_count * WARP_PER_SM;
			this->array_size = this->grid_size * this->block_size;
			this->operation_num = OPS;
		}

		auto start_gen = rad::mysecond();
		this->generate_input();
		auto end_gen = rad::mysecond();

		if (this->parameters.verbose) {
			std::cout << "Input generation time: " << end_gen - start_gen
					<< std::endl;
		}

		start_gen = rad::mysecond();
		//Set the size of
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
		std::uniform_int_distribution<int_t> dist { 1, RANGE_INT_VAL };
		this->input_gold_host.resize(MAX_THREAD_BLOCK);

		for (auto& i : this->input_gold_host)
			i = dist(mersenne_engine);

		this->input_device = this->input_gold_host;
	}

	virtual ~MicroInt() = default;

	void copy_back_output() {
		if (this->parameters.mem_compare_gpu == false) {
			this->output_device.to_vector(this->output_host);
		}
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
		if (parameters.mem_compare_gpu) {
			return this->compare_on_gpu();
		}
		//max num threads on host
		//x8 threads is the number that fits
		auto n_threads = this->array_size / this->input_gold_host.size();
		size_t slice = this->input_gold_host.size();
//		std::cout << "N THREADS " << n_threads << " SLICE " << slice << std::endl;
//		std::vector<std::thread> thread_array;
		std::vector<size_t> error_count_vector(n_threads, 0);

		//the last thread will do slice + rest of the division
#pragma omp parallel for
		for (auto i = 0; i < this->array_size; i += slice) {
//			thread_array.push_back(
//					std::thread(&MicroInt::internal_host_memory_compare, this,
			this->internal_host_memory_compare(&this->input_gold_host[0],
					&this->output_host[i], &error_count_vector[i/slice],
					slice);
		}

//		for (auto& t : thread_array) {
//			t.join();
//		}

		return std::accumulate(error_count_vector.begin(),
				error_count_vector.end(), 0);
	}

	void internal_host_memory_compare(int_t* output_ptr, int_t* gold_ptr,
			size_t* this_thread_error_count, size_t gold_size) {

		std::unique_lock<std::mutex> lock(this->thread_mutex, std::defer_lock);
		for (size_t i = 0; i < gold_size; i++) {
			auto output = output_ptr[i];
			auto gold = gold_ptr[i];
			if (output != gold) {
//					std::string error_detail;
//					error_detail = "position: " + std::to_string(i);
//					error_detail += " e: " + std::to_string(lhs[i]);
//					error_detail += " r: " + std::to_string(rhs[i]);
//					std::cout << error_detail << std::endl;
				*this_thread_error_count++;

//					lock.lock();
//					this->log.log_error_detail(error_detail);
//					lock.unlock();
			}
		}

	}

};

template<>
void MicroInt<int32_t>::execute_micro();
template<>
size_t MicroInt<int32_t>::compare_on_gpu();

#endif /* MICROINT_H_ */
