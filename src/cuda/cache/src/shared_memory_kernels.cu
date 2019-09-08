/*
 * shared_memory_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "Parameters.h"
#include "CacheLine.h"
#include "utils.h"
#include "SharedMemory.h"

template<const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_shared_memory_kernel(const CacheLine<LINE_SIZE> *input,
		CacheLine<LINE_SIZE> *output, const int64 sleep_cycles) {

	__shared__  volatile CacheLine<LINE_SIZE> V[V_SIZE];

	if (threadIdx.x < V_SIZE && blockIdx.y == 0) {
		V[threadIdx.x] = input[blockIdx.x * V_SIZE + threadIdx.x];

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		output[blockIdx.x * V_SIZE + threadIdx.x] = V[threadIdx.x];
	}
}

void SharedMemory::test(const uint32& mem) {
	//	//Set each element of V array

	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<uint64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<CacheLine< CACHE_LINE_SIZE>> input_device_1 =
			this->input_host_1;
	rad::DeviceVector<CacheLine< CACHE_LINE_SIZE>> output_device_1 =
			this->output_host_1;

//	//Set the number of threads
//	//These archs support two blocks per SM with 48KB of shared memory
	switch (this->device) {
	case K20:
	case K40: {
		constexpr uint32 v_size = MAX_KEPLER_SHARED_MEMORY / CACHE_LINE_SIZE;
		test_shared_memory_kernel<v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), cycles);
		break;
	}
	case XAVIER:
	case TITANV: {
		constexpr uint32 v_size = MAX_VOLTA_SHARED_MEMORY / CACHE_LINE_SIZE;
		test_shared_memory_kernel<v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), cycles);
		break;
	}
	}

	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->hit_vector_host = hit_vector_device.to_vector();
	this->miss_vector_host = miss_vector_device.to_vector();
	this->output_host_1 = input_device_1.to_vector();
}

/**
 * Shared memory size is in bytes
 */
SharedMemory::SharedMemory(const Parameters& parameters) :
		Memory<CacheLine<CACHE_LINE_SIZE>>(parameters) {
//This switch is only to set manually the cache line size
//since it is hard to check it at runtime
	uint32 v_size;
	uint32 max_shared_mem;
	switch (parameters.device) {
	case K20:
	case K40:
		max_shared_mem = MAX_KEPLER_SHARED_MEMORY;
		v_size = max_shared_mem / CACHE_LINE_SIZE;

		if (max_shared_mem != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));
		break;
	case XAVIER:
	case TITANV:
		max_shared_mem = MAX_VOLTA_SHARED_MEMORY;
		v_size = max_shared_mem / CACHE_LINE_SIZE;

		if (max_shared_mem * 2 != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));
		break;
	}

	this->threads_per_block = dim3(v_size);
	uint32 v_size_multiple_threads = v_size * parameters.number_of_sms;
	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

std::string SharedMemory::error_detail(uint32 i, uint32 e, uint32 r,
		uint64 hits, uint64 misses, uint64 false_hits) {
	std::string error_detail = "";
	error_detail += " i:" + std::to_string(i);
	error_detail += " cache_line:" + std::to_string(i / CACHE_LINE_SIZE);
	error_detail += " e:" + std::to_string(e);
	error_detail += " r:" + std::to_string(r);
	error_detail += " hits: " + std::to_string(hits);
	error_detail += " misses: " + std::to_string(misses);
	error_detail += " false_hits: " + std::to_string(false_hits);
	return error_detail;
}

void SharedMemory::call_checker(
		const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
		const uint32& valGold, Log& log, uint64 hits, uint64 misses,
		uint64 false_hits, bool verbose) {

	this->check_output_errors((uint32*) v1.data(), valGold, log, hits, misses,
			false_hits, verbose,
			v1.size() * CHUNK_SIZE(CACHE_LINE_SIZE, uint32));
}
