/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "ReadOnly.h"
#include "utils.h"
#include "Parameters.h"
#include "device_functions.h"


__global__ void test_read_only_kernel(
		const __restrict__ uint64* constant_mem_array, uint64 *output_array, uint64 *output_array_aux,
		int64 *hit_array, int64 *miss_array, int64 sleep_cycles) {

	register uint32 tx = (blockIdx.x + blockIdx.y * gridDim.x)
			* (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x)
			+ threadIdx.x;
	int64 miss, hit;
	//ldg is a direct load to const memory
	//first round
	miss = clock64();
	uint64 first_round = __ldg(constant_mem_array + tx);
	miss = clock64() - miss;

	sleep_cuda(sleep_cycles);

	hit = clock64();
	uint64 second_round = __ldg(constant_mem_array + tx);
	hit = clock64() - hit;

	__syncthreads();
	output_array[tx] = second_round;
	hit_array[tx] = hit;
	miss_array[tx] = miss;
	output_array_aux[tx] = first_round;
}

void ReadOny::test(const uint64& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint64> output_device_1 = this->output_host_1;
	rad::DeviceVector<uint64> output_device_aux = this->output_host_1;

	rad::DeviceVector<int64> device_hit = this->hit_vector_host;
	rad::DeviceVector<int64> device_miss = this->miss_vector_host;

	switch (this->device) {
	case K20:
	case K40: {

		test_read_only_kernel<<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), output_device_aux.data(),
				device_hit.data(), device_miss.data(), cycles);
		break;
	}

	case XAVIER:
	case TITANV: {
		test_read_only_kernel<<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), output_device_aux.data(),
				device_hit.data(), device_miss.data(), cycles);
		break;
	}
	}
    rad::checkFrameworkErrors(cudaPeekAtLastError());
    rad::checkFrameworkErrors(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->output_host_1 = output_device_1.to_vector();
	this->hit_vector_host = device_hit.to_vector();
	this->miss_vector_host = device_miss.to_vector();
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

	auto block_max_threads = BLOCK_SIZE * BLOCK_SIZE;
	auto slice = v_size / block_max_threads;
	threads_per_block = dim3(block_max_threads);
	block_size = dim3(slice);

	// Each block with one thread using all read-only cache
	this->input_host_1.resize(v_size);
	this->output_host_1.resize(v_size);
	this->hit_vector_host.resize(v_size);
	this->miss_vector_host.resize(v_size);
}

bool ReadOny::call_checker(uint64& gold, rad::Log& log, int64& hits, int64& misses,
		int64& false_hits, bool verbose) {
	uint64* out_ptr1 = (uint64*) (this->output_host_1.data());
	uint64* out_ptr2 = (uint64*) (this->output_host_2.data());
	uint64* out_ptr3 = (uint64*) (this->output_host_3.data());

	return this->check_output_errors(out_ptr1, out_ptr2, out_ptr3, gold, log,
			hits, misses, false_hits, this->output_host_1.size(), verbose);
}

