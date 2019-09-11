/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "ReadOnly.h"
#include "utils.h"
#include "Parameters.h"

template<uint32 READ_ONLY_MEM_SIZE, uint32 SHARED_PER_SM>
__global__ void test_read_only_kernel(
		__const  __restrict__  uint64* constant_mem_array, uint64 *output_array, int64 sleep_cycles) {

	__shared__ uint64 max_shared_mem[SHARED_PER_SM];
	uint32 tx = blockIdx.x * READ_ONLY_MEM_SIZE + threadIdx.x + threadIdx.y;


	if (tx < READ_ONLY_MEM_SIZE) {

		//ldg is a direct load to const memory
		//first round
		uint64 first_round = __ldg(&constant_mem_array[tx]);
		max_shared_mem[threadIdx.x] = first_round;

		sleep_cuda(sleep_cycles);

		output_array[tx] = max_shared_mem[threadIdx.x];

	}
}

void ReadOny::test(const uint64& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint64> output_device_1 = this->output_host_1;

	switch (this->device) {
	case K20:
	case K40: {
		test_read_only_kernel<MAX_KEPLER_CONSTANT_MEMORY, MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1> <<<
				block_size, threads_per_block>>>(input_device_1.data(), output_device_1.data(), cycles);
		break;
	}

	case XAVIER:
	case TITANV: {
		test_read_only_kernel<MAX_VOLTA_CONSTANT_MEMORY, MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1> <<<
				block_size, threads_per_block>>>(input_device_1.data(), output_device_1.data(), cycles);
	}
	}


	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->output_host_1 = output_device_1.to_vector();
}

ReadOny::ReadOny(const Parameters& parameters) : Memory<uint64>(parameters) {

	uint32 v_size;
	switch (parameters.device) {
	case K20:
	case K40: {
		v_size = MAX_KEPLER_CONSTANT_MEMORY / sizeof(uint64);

		if (MAX_KEPLER_CONSTANT_MEMORY != parameters.const_memory_per_block)
					error(
							"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
									+ std::to_string(parameters.const_memory_per_block));

		threads_per_block = dim3(BLOCK_SIZE * BLOCK_SIZE, v_size / (BLOCK_SIZE * BLOCK_SIZE));
		break;
	}
	case TITANV: {
		v_size = MAX_VOLTA_CONSTANT_MEMORY / sizeof(uint64);

		if (MAX_VOLTA_CONSTANT_MEMORY != parameters.const_memory_per_block)
					error(
							"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
									+ std::to_string(parameters.const_memory_per_block));

		//For Maxwell and above each SM can execute 4 blocks
		threads_per_block = dim3(BLOCK_SIZE * BLOCK_SIZE, v_size / (BLOCK_SIZE * BLOCK_SIZE));
		std::cout << block_size.x << std::endl;
		std::cout << threads_per_block.x << " " << threads_per_block.y <<  std::endl;
		break;
	}
	}

	// Each block with one thread using all read-only cache
	const uint32 v_size_multiple_threads = v_size * parameters.number_of_sms;
	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

