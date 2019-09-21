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

#define NUMBEROFELEMENTS 48
#include "l1_move_function.h"

/*
 * l1_size size of the L1 cache
 * V_size = l1_size / sizeof(CacheLine)
 */
template<const uint32 COUNT, const uint32 SHARED_PER_SM>
__global__ void test_l1_cache_kernel(uint64 *in, uint64 *out, int64 *hits,
		int64 *miss, const int64 sleep_cycles) {

	__shared__ int64 l1_t_hit[SHARED_PER_SM];
	__shared__ int64 l1_t_miss[SHARED_PER_SM];

	const uint64 i = (blockIdx.x * blockDim.x + threadIdx.x) * NUMBEROFELEMENTS;

	uint64 rs[COUNT], rt[COUNT];

	const int64 t1_miss = clock64();
	mov_cache_data(rs, in + i);
//#pragma unroll COUNT
//	for(uint32 k = 0; k < COUNT; k++){
//		rs[k] = in[i + k];
//	}
	l1_t_miss[threadIdx.x] = clock64() - t1_miss;

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	//last checking
	const register int64 t1_hit = clock64();
	mov_cache_data(rt, in + i);
//#pragma unroll COUNT
//	for(uint32 k = 0; k < COUNT; k++){
//		rt[k] = in[i + k];
//	}
	l1_t_hit[threadIdx.x] = clock64() - t1_hit;

	mov_cache_data(out + i, rt);
	mov_cache_data(in + i, rs);
//#pragma unroll COUNT
//	for(uint32 k = 0; k < COUNT; k++){
//		in[i + k] = rs[k];
//		out[i + k] = rt[k];
//	}

	//saving miss and hit
	miss[i] = l1_t_miss[threadIdx.x];
    hits[i] = l1_t_hit[threadIdx.x];
}

L1Cache::L1Cache(const Parameters& parameters) :
		Memory<uint64>(parameters) {
	uint32 v_size;
	switch (device) {
	case K20:
	case K40:
		v_size = MAX_KEPLER_L1_MEMORY / sizeof(uint64);
		break;
	case XAVIER:
	case TITANV:
		v_size = MAX_VOLTA_L1_MEMORY / CACHE_LINE_SIZE;
		break;
	}

//	this->threads_per_block = dim3(v_size);
	this->threads_per_block = dim3(v_size / NUMBEROFELEMENTS);

	uint32 v_size_multiple_threads = v_size * parameters.number_of_sms;
//			* CACHE_LINE_SIZE_BY_INT32; // Each block with one thread using all l1 cache

	this->hit_vector_host.resize(v_size_multiple_threads);
	this->miss_vector_host.resize(v_size_multiple_threads);

	this->input_host_1.resize(v_size_multiple_threads);
	this->output_host_1.resize(v_size_multiple_threads);
}

void L1Cache::test(const uint64& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
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

		test_l1_cache_kernel<NUMBEROFELEMENTS, MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1> <<<
				block_size, threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles);

		break;
	}
	case XAVIER:
	case TITANV: {
		// cache l1 has 128 Kbytes
		//BUT, only 98304 bytes are destined to L1 memory
		//so alloc 98304 bytes
		// cache line has 128 bytes
//		constexpr uint32 v_size = MAX_VOLTA_L1_MEMORY / CACHE_LINE_SIZE;

//		test_l1_cache_kernel<v_size, MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1> <<<
//				block_size, threads_per_block>>>(input_device_1.data(),
//				output_device_1.data(), hit_vector_device.data(),
//				miss_vector_device.data(), cycles);
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
