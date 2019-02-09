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
		std::int64_t sleep_cycles, const byte t) {

	__shared__ CacheLine<LINE_SIZE> V[V_SIZE];

	if (threadIdx.x < V_SIZE && blockIdx.y == 0) {
		V[threadIdx.x] = lines[blockIdx.x * V_SIZE + threadIdx.x];

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		register CacheLine<LINE_SIZE> r = V[threadIdx.x];
		//bitwise operation
		for (uint32 it = 0; it < LINE_SIZE; it++)
			if (r[it] != t)
				atomicAdd(&shared_mem_err, 1);

		lines[blockIdx.x * V_SIZE + threadIdx.x] = r;

	}
	__syncthreads();
}

template<const uint32 V_SIZE, const uint32 SHARED_LINE_SIZE,
		const uint32 SHARED_MEMORY_SIZE>
Tuple test_shared_memory(const uint32 number_of_sms, const byte t_byte,
		const uint64 cycles, dim3& block_size, dim3& threads_per_block) {
	uint32 v_size_multiple_threads = V_SIZE * number_of_sms;
	uint64 shared_mem_err_host = 0;
	cuda_check(
			cudaMemcpyToSymbol(shared_mem_err, &shared_mem_err_host,
					sizeof(uint64), 0));

//Set each element of V array
	CacheLine<SHARED_LINE_SIZE> *V_dev;
	cuda_check(
			cudaMalloc(&V_dev,
					sizeof(CacheLine<SHARED_LINE_SIZE> )
							* v_size_multiple_threads));
	cuda_check(
			cudaMemset(V_dev, t_byte,
					sizeof(CacheLine<SHARED_LINE_SIZE> )
							* v_size_multiple_threads));

	//Set the number of threads
	//These archs support two blocks per SM with 48KB of shared memory
//#if __CUDA_ARCH__ >= 500
//	dim3 block_size(number_of_sms, number_of_sms), threads_per_block(V_SIZE);
//#else
//	dim3 block_size(number_of_sms), threads_per_block(V_SIZE);
//#endif

	test_shared_memory_kernel<V_SIZE, SHARED_LINE_SIZE> <<<block_size,
	threads_per_block, SHARED_MEMORY_SIZE>>>(V_dev, cycles, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(
			cudaMemcpyFromSymbol(&shared_mem_err_host, shared_mem_err,
					sizeof(uint64), 0));

//V array host
	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host(v_size_multiple_threads, 0);
	cuda_check(
			cudaMemcpy(V_host.data(), V_dev,
					sizeof(CacheLine<SHARED_LINE_SIZE> )
							* v_size_multiple_threads, cudaMemcpyDeviceToHost));

	cuda_check(cudaFree(V_dev));

	Tuple t;

	t.cache_lines.assign((byte*) V_host.data(),
			(byte*) V_host.data()
					+ (sizeof(CacheLine<SHARED_LINE_SIZE> ) * V_host.size()));
	t.misses = {};

	t.hits = {};
	t.errors = shared_mem_err_host;
	return t;
}

/**
 * Shared memory size is in bytes
 */
Tuple test_shared_memory(const Parameters& parameters) {
//This switch is only to set manually the cache line size
//since it is hard to check it at runtime
	switch (parameters.device) {
	case K40: {
		const uint32 max_shared_mem = 48 * 1024;

		if (max_shared_mem != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));

		const uint32 cache_line_size = 128;
		const uint32 v_size = max_shared_mem / cache_line_size;
		dim3 block_size(parameters.number_of_sms, 1), threads_per_block(v_size);

		return test_shared_memory<v_size, cache_line_size, max_shared_mem>(
				parameters.number_of_sms, parameters.t_byte,
				parameters.one_second_cycles, block_size, threads_per_block);
//		break;
	}
	case TITANV: {
		const uint32 max_shared_mem = 48 * 1024;

		if (max_shared_mem != parameters.shared_memory_size)
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));

		const uint32 cache_line_size = 128;
		const uint32 v_size = max_shared_mem / cache_line_size;
		dim3 block_size(parameters.number_of_sms, 1), threads_per_block(v_size);

		return test_shared_memory<v_size, cache_line_size, max_shared_mem>(
				parameters.number_of_sms, parameters.t_byte,
				parameters.one_second_cycles, block_size, threads_per_block);
//		break;
	}
	}

	return Tuple();
}

