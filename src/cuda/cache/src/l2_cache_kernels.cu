/*
 * l2_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */
#include <iostream>
#include <vector>
#include "kernels.h"
#include "CacheLine.h"
#include "utils.h"

__device__ uint64 l2_cache_err;

template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_l2_cache_kernel(CacheLine<LINE_SIZE> *lines,
		int_t *l2_hit_array, int_t *l2_miss_array, std::int64_t sleep_cycles,
		byte t) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;

	for (uint32 i = 0; i < V_SIZE; i++) {
		int_t t1 = clock();
		volatile register auto r = lines[tx + i];
		int_t t2 = clock();
		l2_hit_array[tx + i] = t2 - t1;
	}

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	for (uint32 i = 0; i < V_SIZE; i++) {
		//last checking
		int_t t1 = clock();
		volatile register auto r = lines[tx + i];
		int_t t2 = clock();
		l2_miss_array[tx + i] = t2 - t1;

		//bitwise operation
		if ((r ^ t) != 0)
			atomicAdd((unsigned long long*) &l2_cache_err, 1);
	}
}

void test_l2_cache(uint32 number_of_sms, Board device) {
	const byte t_byte = 39;
	const uint32 l2_size = 1536 * 1024; // cache l1 has 65536 bytes
	const uint32 cache_line_size = 128; // size in bytes
	const uint32 v_size = l2_size / cache_line_size; // 12288 lines

	//device arrays
	int32 *l2_hit_array_device, *l2_miss_array_device;
	cudaMalloc(&l2_hit_array_device, sizeof(int32) * v_size);
	cudaMalloc(&l2_miss_array_device, sizeof(int32) * v_size);

	//Host arrays
	std::vector<int32> l1_hit_array_host(v_size), l1_miss_array_host(v_size);

	//Set each element of V array
	CacheLine<cache_line_size> *V_dev;
	std::vector<CacheLine<cache_line_size> > V_host(v_size, t_byte);
//	V_host = new CacheLine<cache_line_size> [v_size];
//	for (int i = 0; i < v_size; i++) {
//		V_host[i] = t_byte;
//	}

	//copy to the gpu
	cudaMalloc(&V_dev, sizeof(CacheLine<cache_line_size> ) * v_size);
	cudaMemcpy(V_dev, V_host.data(),
			sizeof(CacheLine<cache_line_size> ) * v_size,
			cudaMemcpyDeviceToHost);

	test_l2_cache_kernel<int32, v_size, cache_line_size> <<<1, 1>>>(V_dev,
			l2_hit_array_device, l2_miss_array_device, 1000000000, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cudaMemcpy(l1_hit_array_host.data(), l2_hit_array_device,
			sizeof(int32) * v_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(l1_miss_array_host.data(), l2_miss_array_device,
			sizeof(int32) * v_size, cudaMemcpyDeviceToHost);

	uint64 bad = 0;
	for (int i = 0; i < v_size; i++) {
		if ((l1_hit_array_host[i] - l1_miss_array_host[i]) > 0)
			bad++;
	}
	std::cout << "TOTAL BAD " << bad << std::endl;

	cudaFree(l2_hit_array_device);
	cudaFree(l2_miss_array_device);
	cudaFree(V_dev);
}

