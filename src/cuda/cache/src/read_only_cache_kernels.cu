/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "ReadOnly.h"
#include "utils.h"
#include "Parameters.h"

__global__ void test_read_only_kernel(
		const __restrict__ uint64* constant_mem_array, uint64 *output_array,
		int64 sleep_cycles) {

	const register uint32 tx = (blockIdx.x + blockIdx.y * gridDim.x)
			* (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)
			+ threadIdx.x;

	//ldg is a direct load to const memory
	//first round
	const register uint64 first_round = __ldg(constant_mem_array + tx);

	sleep_cuda(sleep_cycles);

	output_array[tx] = first_round;
}

void ReadOny::test(const uint64& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint64> output_device_1 = this->output_host_1;

	switch (this->device) {
	case K20:
	case K40: {

		test_read_only_kernel<<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), cycles);
		break;
	}

	case XAVIER:
	case TITANV: {
		test_read_only_kernel<<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), cycles);
		break;
	}
	}
	cuda_check(cudaPeekAtLastError());
	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->output_host_1 = output_device_1.to_vector();
}

ReadOny::ReadOny(const Parameters& parameters) :
		Memory<uint64>(parameters) {

	uint32 v_size;
	switch (parameters.device) {
	case K20:
	case K40: {
		v_size = MAX_KEPLER_CONSTANT_MEMORY / sizeof(uint64);

		if (MAX_KEPLER_CONSTANT_MEMORY != parameters.const_memory_per_block)
			error(
					"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(
									parameters.const_memory_per_block));
		break;
	}
	case TITANV: {
		v_size = MAX_VOLTA_CONSTANT_MEMORY / sizeof(uint64);

		if (MAX_VOLTA_CONSTANT_MEMORY != parameters.const_memory_per_block)
			error(
					"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(
									parameters.const_memory_per_block));
		break;
	}
	}

	auto const_element_per_block = v_size / (BLOCK_SIZE * BLOCK_SIZE);
	threads_per_block = dim3((BLOCK_SIZE * BLOCK_SIZE));
	block_size = dim3(parameters.number_of_sms, const_element_per_block);

	auto v_size_multiple_threads = v_size * parameters.number_of_sms;
	// Each block with one thread using all read-only cache
	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

