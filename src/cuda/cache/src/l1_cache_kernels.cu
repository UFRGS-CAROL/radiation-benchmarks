/*
 * kernels.cu
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#include "utils.h"
#include "kernels.h"
#include "CacheLine.h"

__device__ uint64 l1_cache_err;

/*
 * l1_size size of the L1 cache
 * V_size = l1_size / sizeof(CacheLine)
 */
template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE,
		const uint32 SHARED_PER_SM>
__global__ void test_l1_cache_kernel(CacheLine<LINE_SIZE> *lines,
		int_t *l1_hit_array, int_t *l1_miss_array, int64 sleep_cycles, byte t) {

	__shared__ int_t l1_t_hit[SHARED_PER_SM / 2];
	__shared__ int_t l1_t_miss[SHARED_PER_SM / 2];
	uint32 i = threadIdx.x;
	if (threadIdx.x < V_SIZE) {

		register CacheLine<LINE_SIZE> r;
		volatile int_t t1 = clock();
		r = lines[blockIdx.x * V_SIZE + i];
		volatile int_t t2 = clock();
		l1_t_miss[i] = t2 - t1;
		lines[blockIdx.x * V_SIZE + i] = r;

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//last checking
		t1 = clock();
		register CacheLine<LINE_SIZE> r2 = lines[blockIdx.x * V_SIZE + i];
		t2 = clock();
		l1_t_hit[i] = t2 - t1;
		lines[blockIdx.x * V_SIZE + i] = r2;

		if (r != t) {
			atomicAdd(&l1_cache_err, 1);
		}
	}

	__syncthreads();
	if (threadIdx.x < V_SIZE) {
		for (uint32 i = 0; i < V_SIZE; i++) {
			l1_miss_array[blockIdx.x * V_SIZE + i] = l1_t_miss[i];
			l1_hit_array[blockIdx.x * V_SIZE + i] = l1_t_hit[i];
		}
	}
}

template<const uint32 V_SIZE, const uint32 L1_LINE_SIZE,
		const uint32 SHARED_PER_SM>
Tuple test_l1_cache(const uint32 number_of_sms, const byte t_byte,
		const int64 cycles) {

	const uint32 v_size_multiple_threads = V_SIZE * number_of_sms; // Each block with one thread using all l1 cache

	//device arrays
	int32 *l1_hit_array_device, *l1_miss_array_device;
	cuda_check(
			cudaMalloc(&l1_hit_array_device,
					sizeof(int32) * v_size_multiple_threads));
	cuda_check(
			cudaMalloc(&l1_miss_array_device,
					sizeof(int32) * v_size_multiple_threads));

	//Set each element of V array
	CacheLine < L1_LINE_SIZE > *V_dev;
	std::vector<CacheLine<L1_LINE_SIZE> > V_host(v_size_multiple_threads);

	for (int i = 0; i < v_size_multiple_threads; i++) {
		V_host[i] = t_byte;
	}

	//copy to the GPU
	cuda_check(
			cudaMalloc(&V_dev,
					sizeof(CacheLine<L1_LINE_SIZE> )
							* v_size_multiple_threads));

	cuda_check(
			cudaMemcpy(V_dev, V_host.data(),
					sizeof(CacheLine<L1_LINE_SIZE> ) * v_size_multiple_threads,
					cudaMemcpyHostToDevice));

	//Set to zero err_check
	uint64 l1_cache_err_host = 0;
	cuda_check(
			cudaMemcpyToSymbol(l1_cache_err, &l1_cache_err_host, sizeof(uint64),
					0));

	dim3 block_size(number_of_sms), threads_per_block(V_SIZE);

	test_l1_cache_kernel<int32, V_SIZE, L1_LINE_SIZE, SHARED_PER_SM> <<<
	block_size, threads_per_block>>>(V_dev, l1_hit_array_device,
			l1_miss_array_device, cycles, t_byte);
	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	std::vector<int32> l1_hit_array_host(v_size_multiple_threads),
			l1_miss_array_host(v_size_multiple_threads);
	cuda_check(
			cudaMemcpy(l1_hit_array_host.data(), l1_hit_array_device,
					sizeof(int32) * v_size_multiple_threads,
					cudaMemcpyDeviceToHost));
	cuda_check(
			cudaMemcpy(l1_miss_array_host.data(), l1_miss_array_device,
					sizeof(int32) * v_size_multiple_threads,
					cudaMemcpyDeviceToHost));
	cuda_check(
			cudaMemcpy(V_host.data(), V_dev,
					sizeof(CacheLine<L1_LINE_SIZE> ) * v_size_multiple_threads,
					cudaMemcpyDeviceToHost));

	cuda_check(
			cudaMemcpyFromSymbol(&l1_cache_err_host, l1_cache_err,
					sizeof(uint64), 0));

	cuda_check(cudaFree(l1_hit_array_device));
	cuda_check(cudaFree(l1_miss_array_device));
	cuda_check(cudaFree(V_dev));

	Tuple t;

	t.cache_lines.assign((byte*) V_host.data(),
			(byte*) V_host.data()
					+ (sizeof(CacheLine<L1_LINE_SIZE> ) * V_host.size()));
	t.misses = std::move(l1_miss_array_host);

	t.hits = std::move(l1_hit_array_host);
	t.errors = l1_cache_err_host;

	return t;
}

Tuple test_l1_cache(const Parameters& parameters) {
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	switch (parameters.device) {
	case K40: {
		// cache l1 has 65536 bytes
		//BUT, only 48kb are destined to L1 memory
		//so alloc 49152 bytes
		// cache line has 128 bytes
		//to force alloc maximum shared memory
		std::cout << "KEPLER\n";
		const uint32 max_l1_cache = 48 * 1024; //bytes
		const uint32 max_shared_mem = 8 * 1024;
		const uint32 cache_line_size = 128;
		const uint32 v_size = max_l1_cache / cache_line_size;
		return test_l1_cache<v_size, cache_line_size, max_shared_mem>(
				parameters.number_of_sms, parameters.t_byte,
				parameters.one_second_cycles);
//		break;
	}
	case TITANV: {
		// cache l1 has 128 Kbytes
		//BUT, only 98304 bytes are destined to L1 memory
		//so alloc 98304 bytes
		// cache line has 128 bytes
		const uint32 max_l1_cache = 96 * 1024; //bytes
		const uint32 max_shared_mem = 8 * 1024;
		const uint32 cache_line_size = 128;
		const uint32 v_size = max_l1_cache / cache_line_size;

		return test_l1_cache<v_size, cache_line_size, max_shared_mem>(
				parameters.number_of_sms, parameters.t_byte,
				parameters.one_second_cycles);
//		break;
	}
	}

	return Tuple();
}

