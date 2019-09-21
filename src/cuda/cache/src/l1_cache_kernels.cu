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

//#include "l1_move_function.h"
#if __CUDA_ARCH__ <= 350
#define SHARED_PER_SM MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1
#elif __CUDA_ARCH__ == 700
#define SHARED_PER_SM MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1
#else
#error CUDA ARCH NOT SPECIFIED.
#endif

__device__ __forceinline__
void mov_cache_data(uint64* dst, uint64* src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
	dst[8] = src[8];
	dst[9] = src[9];
	dst[10] = src[10];
	dst[11] = src[11];
	dst[12] = src[12];
	dst[13] = src[13];
	dst[14] = src[14];
	dst[15] = src[15];
}

__global__ void test_l1_cache_kernel(uint64 *in, uint64 *out, int64 *hits,
		int64 *miss, const int64 sleep_cycles) {

	__shared__ int64 l1_t_hit[SHARED_PER_SM];
	__shared__ int64 l1_t_miss[SHARED_PER_SM];

	const uint32 index = blockIdx.x * blockDim.x + threadIdx.x;;
	const uint32 i = index * CACHE_LINE_SIZE_BY_INT64;

	uint64 rs[CACHE_LINE_SIZE_BY_INT64], rt[CACHE_LINE_SIZE_BY_INT64];

	const int64 t1_miss = clock64();
	mov_cache_data(rt, in + i);
	l1_t_miss[threadIdx.x] = clock64() - t1_miss;

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	//last checking
	const int64 t1_hit = clock64();
	mov_cache_data(rs, in + i);
	l1_t_hit[threadIdx.x] = clock64() - t1_hit;

	mov_cache_data(out + i, rs);
	mov_cache_data(in + i, rt);

//saving miss and hit
	miss[index] = l1_t_miss[threadIdx.x];
	hits[index] = l1_t_hit[threadIdx.x];
}

L1Cache::L1Cache(const Parameters& parameters) :
		Memory<uint64>(parameters) {
	uint32 v_size;
	switch (device) {
	case K20:
	case K40:
		v_size = MAX_KEPLER_L1_MEMORY / CACHE_LINE_SIZE;
		break;
	case XAVIER:
	case TITANV:
		v_size = MAX_VOLTA_L1_MEMORY / sizeof(uint64);
		break;
	}

	this->threads_per_block = dim3(v_size);
	// Each block with one thread using all l1 cache
	uint32 total_size = v_size * parameters.number_of_sms;

	uint32 v_size_multiple_threads = total_size	* CACHE_LINE_SIZE_BY_INT64;

	std::cout << "BLOCK SIZE " << this->threads_per_block.x << "x"
			<< this->threads_per_block.y << std::endl;
	std::cout << "GRID SIZE " << this->block_size.x << "x" << this->block_size.y
			<< std::endl;
	std::cout << "TOTAL SIZE " << v_size_multiple_threads << std::endl;

	this->hit_vector_host.resize(total_size);
	this->miss_vector_host.resize(total_size);

	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

void L1Cache::test(const uint64& mem) {
//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	std::fill(this->hit_vector_host.begin(), this->hit_vector_host.end(), 0);
	std::fill(this->miss_vector_host.begin(), this->miss_vector_host.end(), 0);

	rad::DeviceVector<int64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<int64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<uint64> input_device_1(this->input_host_1);
	rad::DeviceVector<uint64> output_device_1(this->output_host_1);

//This switch is only to set manually the cache line size
//since it is hard to check it at runtime
	switch (device) {
	case K20:
	case K40: {
		// cache l1 has 65536 bytes
		//BUT, only 48kb are destined to L1 memory
		//so alloc 49152 bytes
		// cache line has 128 bytes
		//to force alloc maximum shared memory
//		constexpr uint32 v_size = MAX_KEPLER_L1_MEMORY / CACHE_LINE_SIZE;

		test_l1_cache_kernel<<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(),
				hit_vector_device.data(), miss_vector_device.data(), cycles);

		break;
	}
	case XAVIER:
	case TITANV: {
		// cache l1 has 128 Kbytes
		//BUT, only 98304 bytes are destined to L1 memory
		//so alloc 98304 bytes
		// cache line has 128 bytes
//		constexpr uint32 v_size = MAX_VOLTA_L1_MEMORY / CACHE_LINE_SIZE;

//		test_l1_cache_kernel<MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1> <<<block_size,
//				threads_per_block>>>(input_device_1.data(),
//				output_device_1.data(), hit_vector_device.data(),
//				miss_vector_device.data(), cycles, mem);
		break;
	}
	}

	cuda_check(cudaPeekAtLastError());
	cuda_check(cudaDeviceSynchronize());
//Host arrays
//Copy back to the host
	this->hit_vector_host = hit_vector_device.to_vector();
	this->miss_vector_host = miss_vector_device.to_vector();
	this->output_host_1 = output_device_1.to_vector();
}
