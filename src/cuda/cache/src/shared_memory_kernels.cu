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
//		auto register r = V[i];
		//bitwise operation
//		if ((r ^ t) != 0)
//			atomicAdd(&shared_mem_err, 1);
	}
}

template<uint32 SHARED_MEMORY_SIZE, uint32 SHARED_LINE_SIZE>
void test_shared_memory(const uint32 number_of_sms, const byte t_byte,
		const uint64 cycles) {
	cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
	const uint32 v_size = SHARED_MEMORY_SIZE / SHARED_LINE_SIZE; // 384 lines

	uint64 mem = 0;
	cuda_check(cudaMemcpyToSymbol(shared_mem_err, &mem, sizeof(uint64), 0));

	//Set each element of V array
	CacheLine<SHARED_LINE_SIZE> *V_dev;
	cuda_check(
			cudaMalloc(&V_dev, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size));
	cuda_check(
			cudaMemset(V_dev, t_byte,
					sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size));

	test_shared_memory_kernel<v_size, SHARED_LINE_SIZE> <<<1, 1>>>(V_dev,
			cycles, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(cudaMemcpyFromSymbol(&mem, shared_mem_err, sizeof(uint64), 0));

	std::cout << "TOTAL BAD " << mem << std::endl;
	cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferNone));
	cuda_check(cudaFree(V_dev));
}

/**
 * Shared memory size is in bytes
 */
Tuple test_shared_memory(const Parameters& parameters) {

	for (byte t_byte : { 0x00, 0xff }) {
		switch (parameters.device) {
		case K40: {

			break;
		}
		case TITANV: {
			break;
		}
		}
	}

	return Tuple();
}

