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

template<const uint32 V_SIZE>
__global__ void test_shared_memory_kernel(uint64 *input, uint64 *output,
		const int64 sleep_cycles) {

	__shared__ uint64 V[V_SIZE * CACHE_LINE_SIZE_BY_INT64];
	const register uint64 index = (blockIdx.x * V_SIZE + threadIdx.x) * CACHE_LINE_SIZE_BY_INT64;

	move_cache_line(V + threadIdx.x, input + index);

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	move_cache_line(output + index, V + threadIdx.x);
}

void SharedMemory::test(const uint64& mem) {
	//	//Set each element of V array

	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint64> output_device_1 = this->output_host_1;

	//Set the number of threads
	//These archs support two blocks per SM with 48KB of shared memory
	switch (this->device) {
	case K20:
	case K40: {
		constexpr uint32 v_size = MAX_KEPLER_SHARED_MEMORY / CACHE_LINE_SIZE;
		test_shared_memory_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), cycles);
		break;
	}
	case XAVIER:
	case TITANV: {
		constexpr uint32 v_size = MAX_VOLTA_SHARED_MEMORY / CACHE_LINE_SIZE;
		test_shared_memory_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), cycles);
		break;
	}
	}

	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->output_host_1 = output_device_1.to_vector();
}

/**
 * Shared memory size is in bytes
 */
SharedMemory::SharedMemory(const Parameters& parameters) :
		Memory<uint64>(parameters) {
//This switch is only to set manually the cache line size
//since it is hard to check it at runtime
	uint32 v_size;
	switch (parameters.device) {
	case K20:
	case K40:
		v_size = MAX_KEPLER_SHARED_MEMORY / CACHE_LINE_SIZE;

		if (MAX_KEPLER_SHARED_MEMORY != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));
		break;
	case XAVIER:
	case TITANV:
		v_size = MAX_VOLTA_SHARED_MEMORY / CACHE_LINE_SIZE;

		if (MAX_VOLTA_SHARED_MEMORY * 2 != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));
		break;
	}

	this->threads_per_block = dim3(v_size);
	uint32 v_size_multiple_threads = v_size * parameters.number_of_sms
			* CACHE_LINE_SIZE_BY_INT64;
	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

bool SharedMemory::call_checker(uint64& gold, Log& log, int64& hits,
		int64& misses, int64& false_hits) {
	uint64* out_ptr = (uint64*) (this->output_host_1.data());
	return this->check_output_errors(out_ptr, gold, log, hits, misses,
			false_hits, this->output_host_1.size());
}
