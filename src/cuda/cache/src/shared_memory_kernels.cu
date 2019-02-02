/*
 * shared_memory_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "kernels.h"
#include "CacheLine.h"
#include "utils.h"

#include <iostream>

__device__ uint64 shared_mem_err;

template<const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_shared_memory_kernel(CacheLine<LINE_SIZE> *lines,
		std::int64_t sleep_cycles, const register byte t) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ CacheLine<LINE_SIZE> V[V_SIZE];

	for (uint32 i = 0; i < V_SIZE; i++) {
		V[i] = lines[tx + i];
	}

	//wait for exposition to neutrons
	sleep_cuda(sleep_cycles);

	for (uint32 i = 0; i < V_SIZE; i++) {
		auto register r = V[i];
		//bitwise operation
		if ((r ^ t) != 0)
			atomicAdd(&shared_mem_err, 1);
	}
}

void test_shared_memory(uint32 number_of_sms, Board device) {
	cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

	const byte t_byte = 39;
	const uint32 shared_memory_size = 48 * 1024; // shared memory has 49152 bytes
	const uint32 shared_line_size = 128; // size in bytes
	const uint32 v_size = shared_memory_size / shared_line_size; // 384 lines

	uint64 mem = 0;
	cuda_check(cudaMemcpyToSymbol(shared_mem_err, &mem, sizeof(uint64), 0));

	//Set each element of V array
	CacheLine<shared_line_size> *V_dev;
	cuda_check(
			cudaMalloc(&V_dev, sizeof(CacheLine<shared_line_size> ) * v_size));
	cuda_check(
			cudaMemset(V_dev, t_byte,
					sizeof(CacheLine<shared_line_size> ) * v_size));

	test_shared_memory_kernel<v_size, shared_line_size> <<<1, 1>>>(V_dev,
			1000000000, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(cudaMemcpyFromSymbol(&mem, shared_mem_err, sizeof(uint64), 0));

	std::cout << "TOTAL BAD " << mem << std::endl;
	cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));
	cuda_check(cudaFree(V_dev));
}

