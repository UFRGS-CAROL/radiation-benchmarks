/*
 * kernels.cu
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

#include "utils.h"
#include "Memory.h"
#include "L1Cache.h"

#if __CUDA_ARCH__ <= 350
#define SHARED_PER_SM MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1
#elif __CUDA_ARCH__ == 700
#define SHARED_PER_SM MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1
#else
#error CUDA ARCH NOT SPECIFIED.
#endif

//#include "l1_move_function.h"

__global__ void test_l1_cache_kernel(cacheline *src, cacheline *dst,
		int64 *hits, int64 *miss, const int64 sleep_cycles) {

	__shared__ int64 t_hits[SHARED_PER_SM];
	__shared__ int64 t_miss[SHARED_PER_SM];

	const uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
	cacheline rm, rh;

	__syncthreads();

	t_miss[threadIdx.x] = clock64();
	rm = src[i];
	t_miss[threadIdx.x] = clock64() - t_miss[threadIdx.x];

	sleep_cuda(sleep_cycles);

	t_hits[threadIdx.x] = clock64();
	rh = src[i];
	t_hits[threadIdx.x] = clock64() - t_hits[threadIdx.x];

	__syncthreads();

	hits[i] = t_hits[threadIdx.x];
	miss[i] = t_miss[threadIdx.x];
	src[i] = rm;
	dst[i] = rh;
}

L1Cache::L1Cache(const Parameters& parameters) :
		Memory<cacheline>(parameters) {
	uint32 v_size;
	switch (device) {
	case K20:
	case K40:
		v_size = MAX_KEPLER_L1_MEMORY / CACHE_LINE_SIZE;
		break;
	case XAVIER:
	case TITANV:
		v_size = MAX_VOLTA_L1_MEMORY / CACHE_LINE_SIZE;
		break;
	}

	this->threads_per_block = dim3(v_size);
	// Each block with one thread using all l1 cache
	uint32 total_size = v_size * parameters.number_of_sms;

	std::cout << "BLOCK SIZE " << this->threads_per_block.x << "x"
			<< this->threads_per_block.y << std::endl;
	std::cout << "GRID SIZE " << this->block_size.x << "x" << this->block_size.y
			<< std::endl;
	std::cout << "TOTAL SIZE " << total_size << std::endl;

	this->hit_vector_host.resize(total_size);
	this->miss_vector_host.resize(total_size);

	this->input_host_1.resize(total_size);
	this->output_host_1.resize(total_size);
}

void L1Cache::test(const uint64& mem) {
	cacheline cl;
	for (auto& l : cl.line) {
		l = mem;
	}

//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), cl);
	std::fill(this->hit_vector_host.begin(), this->hit_vector_host.end(), 0);
	std::fill(this->miss_vector_host.begin(), this->miss_vector_host.end(), 0);

	rad::DeviceVector<int64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<int64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<cacheline> input_device_1(this->input_host_1);
	rad::DeviceVector<cacheline> output_device_1(this->output_host_1);

	test_l1_cache_kernel<<<block_size, threads_per_block>>>(
			input_device_1.data(), output_device_1.data(),
			hit_vector_device.data(), miss_vector_device.data(), cycles);

	cuda_check(cudaPeekAtLastError());
	cuda_check(cudaDeviceSynchronize());
//Host arrays
//Copy back to the host
	this->hit_vector_host = hit_vector_device.to_vector();
	this->miss_vector_host = miss_vector_device.to_vector();
	this->output_host_1 = output_device_1.to_vector();
}

bool L1Cache::call_checker(uint64& gold, rad::Log& log, int64& hits, int64& misses,
		int64& false_hits, bool verbose) {

	return this->check_output_errors((uint64*) (this->output_host_1.data()),
			(uint64*) (this->output_host_2.data()),
			(uint64*) (this->output_host_3.data()), gold, log, hits, misses,
			false_hits, this->output_host_1.size() * CACHE_LINE_SIZE_BY_INT64, verbose);
}
