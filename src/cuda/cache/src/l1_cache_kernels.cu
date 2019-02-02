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
template<typename int_t, const uint32 V_SIZE,
		const uint32 LINE_SIZE>
__global__ void test_l1_cache_kernel(CacheLine<LINE_SIZE> *lines,
		int_t *l1_hit_array, int_t *l1_miss_array, std::int64_t sleep_cycles,
		byte t) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int_t l1_t_hit[V_SIZE];
	__shared__ int_t l1_t_miss[V_SIZE];

	for (uint32 i = 0; i < V_SIZE; i++) {
		int_t t1 = clock();
		register auto r = lines[tx + i];
		int_t t2 = clock();
		l1_t_miss[i] = t2 - t1;
//		lines[tx + i] = r;
	}

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	for (uint32 i = 0; i < V_SIZE; i++) {
		//last checking
		int_t t1 = clock();
		register auto r = lines[tx + i];
		int_t t2 = clock();
		l1_t_hit[i] = t2 - t1;

		//bitwise operation
		if ((r ^ t) != 0)
			atomicAdd((unsigned long long*)&l1_cache_err, 1);

//		//saving the result
		l1_hit_array[tx + i] = l1_t_hit[i];
		l1_miss_array[tx + i] = l1_t_miss[i];
	}

}

std::vector<std::string> test_l1_cache(uint32 number_of_sms, Board device) {
	std::vector<std::string> errors;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	const byte t_byte = 39;
	const uint32 l1_size = 64 * 1024; // cache l1 has 65536 bytes
	const uint32 cache_line_size = 128; // size in bytes
	const uint32 v_size = l1_size / cache_line_size; // 512 lines

	//device arrays
	int32 *l1_hit_array_device, *l1_miss_array_device;
	cudaMalloc(&l1_hit_array_device, sizeof(int32) * v_size);
	cudaMalloc(&l1_miss_array_device, sizeof(int32) * v_size);

	//Host arrays
	int32 *l1_hit_array_host = new int32[v_size];
	int32 *l1_miss_array_host = new int32[v_size];

	//Set each element of V array
	CacheLine<cache_line_size> *V_dev, *V_host;
	V_host = new CacheLine<cache_line_size> [v_size];
	for (int i = 0; i < v_size; i++) {
		V_host[i] = t_byte;
	}

	//copy to the gpu
	cudaMalloc(&V_dev, sizeof(CacheLine<cache_line_size> ) * v_size);
	cudaMemcpy(V_dev, V_host, sizeof(CacheLine<cache_line_size> ) * v_size,
			cudaMemcpyDeviceToHost);

	test_l1_cache_kernel<int32, v_size, cache_line_size> <<<1, 1>>>(V_dev,
			l1_hit_array_device, l1_miss_array_device, 1000000000, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cudaMemcpy(l1_hit_array_host, l1_hit_array_device,
			sizeof(int32) * v_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(l1_miss_array_host, l1_miss_array_device,
			sizeof(int32) * v_size, cudaMemcpyDeviceToHost);
	auto bad = 0;
	for (int i = 0; i < v_size; i++) {
		if ((l1_hit_array_host[i] - l1_miss_array_host[i]) > 0)
			bad++;
	}
	std::cout << "TOTAL BAD " << bad << std::endl;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

	cudaFree(l1_hit_array_device);
	cudaFree(l1_miss_array_device);
	cudaFree(V_dev);
	delete[] V_host;
	delete[] l1_hit_array_host;
	delete[] l1_miss_array_host;
	return errors;
}

std::vector<std::string> test_l1_cache(const Parameters& parameters){
	std::vector<std::string> errors;

	return errors;
}
