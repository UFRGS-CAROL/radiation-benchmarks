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
template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_l1_cache_kernel(CacheLine<LINE_SIZE> *lines,
		int_t *l1_hit_array, int_t *l1_miss_array, int64 sleep_cycles, byte t) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x == 0) {
		__shared__ int_t l1_t_hit[V_SIZE];
		__shared__ int_t l1_t_miss[V_SIZE];

		for (uint32 i = 0; i < V_SIZE; i++) {
			volatile int_t t1 = clock();
			volatile auto r = lines[tx + i];
			volatile int_t t2 = clock();
			l1_t_miss[i] = t2 - t1;
		}

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		for (uint32 i = 0; i < V_SIZE; i++) {
			//last checking
			volatile int_t t1 = clock();
			volatile auto r = lines[tx + i];
			volatile int_t t2 = clock();
			l1_t_hit[i] = t2 - t1;

			//bitwise operation
			if ((r ^ t) != 0)
				atomicAdd((unsigned long long*) &l1_cache_err, 1);

//		//saving the result
			l1_hit_array[tx + i] = l1_t_hit[i];
			l1_miss_array[tx + i] = l1_t_miss[i];
		}

	}
	__syncthreads();
}

template<uint32 L1_MEMORY_SIZE, uint32 L1_LINE_SIZE>
std::vector<std::string> test_l1_cache(const uint32 number_of_sms,
		const byte t_byte, const int64 cycles) {
	std::vector<std::string> errors;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	const uint32 v_size = L1_MEMORY_SIZE / L1_LINE_SIZE; // 512 lines

	const uint32 v_size_multiple_threads = v_size * number_of_sms; // Each block with one thread using all l1 cache

	//device arrays
	int32 *l1_hit_array_device, *l1_miss_array_device;
	cuda_check(
			cudaMalloc(&l1_hit_array_device,
					sizeof(int32) * v_size_multiple_threads));
	cuda_check(
			cudaMalloc(&l1_miss_array_device,
					sizeof(int32) * v_size_multiple_threads));

	//Set each element of V array
	CacheLine<L1_LINE_SIZE> *V_dev;
	std::vector<CacheLine<L1_LINE_SIZE> > V_host(v_size_multiple_threads);

	for (int i = 0; i < v_size_multiple_threads; i++) {
		V_host[i] = t_byte;
	}

	//copy to the gpu
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
	copy_to_gpu<uint64>("l1_cache_err", l1_cache_err_host);

	test_l1_cache_kernel<int32, v_size, L1_LINE_SIZE> <<<number_of_sms,
	BLOCK_SIZE>>>(V_dev, l1_hit_array_device, l1_miss_array_device, cycles,
			t_byte);
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

//	for (auto i = 0; i < v_size_multiple_threads; i++) {
//		if ((l1_hit_array_host[i] - l1_miss_array_host[i]) > 0)
//			bad++;
//	}
	l1_cache_err_host = copy_from_gpu<uint64>("l1_cache_err");

	std::cout << "TOTAL BAD " << l1_cache_err_host << std::endl;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

	cuda_check(cudaFree(l1_hit_array_device));
	cuda_check(cudaFree(l1_miss_array_device));
	cuda_check(cudaFree(V_dev));

	return errors;
}

std::vector<std::string> test_l1_cache(const Parameters& parameters) {
	std::vector<std::string> errors;
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	switch (parameters.device) {
	case K40: {
		// cache l1 has 65536 bytes
		// cache line has 128 bytes
		test_l1_cache<65536, 128>(parameters.number_of_sms, 0xff,
				parameters.one_second_cycles);
		break;
	}
	case TITANV: {
		break;
	}
	}
	return errors;
}
