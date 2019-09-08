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
#include "CacheLine.h"
#include "Memory.h"
#include "L1Cache.h"

/*
 * l1_size size of the L1 cache
 * V_size = l1_size / sizeof(CacheLine)
 */
template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE,
		const uint32 SHARED_PER_SM>
__global__ void test_l1_cache_kernel(CacheLine<LINE_SIZE> *input_lines,
		CacheLine<LINE_SIZE> *output_lines, int_t *l1_hit_array,
		int_t *l1_miss_array, int64 sleep_cycles, uint32 t) {

	__shared__ int_t l1_t_hit[SHARED_PER_SM / 2];
	__shared__ int_t l1_t_miss[SHARED_PER_SM / 2];

	if (threadIdx.x < V_SIZE && blockIdx.y == 0) {
		const int_t t1_miss = clock();
		volatile CacheLine<LINE_SIZE> r = input_lines[blockIdx.x * V_SIZE
				+ threadIdx.x];
		const int_t t2_miss = clock();
		l1_t_miss[threadIdx.x] = t2_miss - t1_miss;

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//last checking
		const int_t t1_hit = clock();
		volatile CacheLine<LINE_SIZE> rt = input_lines[blockIdx.x * V_SIZE + threadIdx.x];
		const int_t t2_hit = clock();
		l1_t_hit[threadIdx.x] = t2_hit - t1_hit;

		l1_miss_array[blockIdx.x * V_SIZE + threadIdx.x] =
				l1_t_miss[threadIdx.x];
		l1_hit_array[blockIdx.x * V_SIZE + threadIdx.x] = l1_t_hit[threadIdx.x];

		//triplication
		output_lines[blockIdx.x * V_SIZE + threadIdx.x] = r;
		rt &= t;
		input_lines[blockIdx.x * V_SIZE + threadIdx.x] = rt;
	}
}

void L1Cache::call_checker(const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
		const uint32& valGold, Log& log, uint64 hits, uint64 misses,
		uint64 false_hits, bool verbose) {

	this->check_output_errors((uint32*) v1.data(), valGold, log, hits, misses,
			false_hits, verbose,
			v1.size() * CHUNK_SIZE(CACHE_LINE_SIZE, uint32));
}

L1Cache::L1Cache(const Parameters& parameters) :
		Memory<CacheLine<CACHE_LINE_SIZE> >(parameters) {
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
	uint32 v_size_multiple_threads = v_size * parameters.number_of_sms; // Each block with one thread using all l1 cache

	this->input_host_1.resize(v_size_multiple_threads);
	this->hit_vector_host.resize(v_size_multiple_threads);
	this->miss_vector_host.resize(v_size_multiple_threads);

	this->output_host_1.resize(v_size_multiple_threads);
}

void L1Cache::test(const uint32& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<uint64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<CacheLine<CACHE_LINE_SIZE>> input_device_1 =
			this->input_host_1;
	rad::DeviceVector<CacheLine<CACHE_LINE_SIZE>> output_device_1 =
			this->output_host_1;

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
		constexpr uint32 v_size = MAX_KEPLER_L1_MEMORY / CACHE_LINE_SIZE;

		test_l1_cache_kernel<uint64, v_size, CACHE_LINE_SIZE,
		MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1> <<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles, mem);

		break;
	}
	case XAVIER:
	case TITANV: {
		// cache l1 has 128 Kbytes
		//BUT, only 98304 bytes are destined to L1 memory
		//so alloc 98304 bytes
		// cache line has 128 bytes
		constexpr uint32 v_size = MAX_VOLTA_L1_MEMORY / CACHE_LINE_SIZE;

		test_l1_cache_kernel<uint64, v_size, CACHE_LINE_SIZE,
		MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1> <<<block_size, threads_per_block>>>(
				input_device_1.data(), output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles, mem);
		break;
	}
	}

	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->hit_vector_host = hit_vector_device.to_vector();
	this->miss_vector_host = miss_vector_device.to_vector();
	this->output_host_1 = input_device_1.to_vector();

//	this->output_host_1[33] = this->output_host_2[33] = this->output_host_3[33] = CacheLine<CACHE_LINE_SIZE>(byte(33));
}

std::string L1Cache::error_detail(uint32 i, uint32 e, uint32 r, uint64 hits,
		uint64 misses, uint64 false_hits) {
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
