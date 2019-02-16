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

//__device__ uint64 shared_mem_err1;
//__device__ uint64 shared_mem_err2;
//__device__ uint64 shared_mem_err3;

template<const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_shared_memory_kernel(CacheLine<LINE_SIZE> *lines1, CacheLine<LINE_SIZE> *lines2, CacheLine<LINE_SIZE> *lines3, std::int64_t sleep_cycles, const byte t) {

	__shared__ CacheLine<LINE_SIZE> V[V_SIZE];

	if (threadIdx.x < V_SIZE && blockIdx.y == 0) {
		V[threadIdx.x] = t; //lines[blockIdx.x * V_SIZE + threadIdx.x];

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//bitwise operation
		//for (uint32 it = 0; it < LINE_SIZE; it++)
		//	if (V[threadIdx.x][it] != t){
		//		atomicAdd(&shared_mem_err1, 1);
                //                atomicAdd(&shared_mem_err2, 1);
                //                atomicAdd(&shared_mem_err3, 1);
                //       }
		lines1[blockIdx.x * V_SIZE + threadIdx.x] = V[threadIdx.x];
		lines2[blockIdx.x * V_SIZE + threadIdx.x] = V[threadIdx.x];
		lines3[blockIdx.x * V_SIZE + threadIdx.x] = V[threadIdx.x];

	}
	__syncthreads();
}

template<const uint32 V_SIZE, const uint32 SHARED_LINE_SIZE, const uint32 SHARED_MEMORY_SIZE>
Tuple test_shared_memory(const uint32 number_of_sms, const byte t_byte, const uint64 cycles, dim3& block_size, dim3& threads_per_block) {

        uint32 v_size_multiple_threads = V_SIZE * number_of_sms;
        
        //set errors counters
	//uint64 shared_mem_err_host1 = 0;
	//uint64 shared_mem_err_host2 = 0;
	//uint64 shared_mem_err_host3 = 0;

	//cuda_check(cudaMemcpyToSymbol(shared_mem_err1, &shared_mem_err_host1, sizeof(uint64), 0));
        //cuda_check(cudaMemcpyToSymbol(shared_mem_err2, &shared_mem_err_host2, sizeof(uint64), 0));
        //cuda_check(cudaMemcpyToSymbol(shared_mem_err3, &shared_mem_err_host3, sizeof(uint64), 0));

        //Set each element of V array
	CacheLine<SHARED_LINE_SIZE> *V_dev1, *V_dev2, *V_dev3;
	cuda_check(cudaMalloc(&V_dev1, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads));
	cuda_check(cudaMalloc(&V_dev2, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads));
	cuda_check(cudaMalloc(&V_dev3, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads));
	
        
	//Set the number of threads
	//These archs support two blocks per SM with 48KB of shared memory
	test_shared_memory_kernel<V_SIZE, SHARED_LINE_SIZE> <<<block_size, threads_per_block>>>(V_dev1, V_dev2, V_dev3, cycles, t_byte);
	cuda_check(cudaDeviceSynchronize());

	//cuda_check(cudaMemcpyFromSymbol(&shared_mem_err_host1, shared_mem_err1, sizeof(uint64), 0));
	//cuda_check(cudaMemcpyFromSymbol(&shared_mem_err_host2, shared_mem_err2, sizeof(uint64), 0));
	//cuda_check(cudaMemcpyFromSymbol(&shared_mem_err_host3, shared_mem_err3, sizeof(uint64), 0));


        //V array host
	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host1(v_size_multiple_threads, 0);
	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host2(v_size_multiple_threads, 0);
	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host3(v_size_multiple_threads, 0);
        
	cuda_check(cudaMemcpy(V_host1.data(), V_dev1, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(V_host2.data(), V_dev2, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads, cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(V_host3.data(), V_dev3, sizeof(CacheLine<SHARED_LINE_SIZE> ) * v_size_multiple_threads, cudaMemcpyDeviceToHost));

	cuda_check(cudaFree(V_dev1));
	cuda_check(cudaFree(V_dev2));
	cuda_check(cudaFree(V_dev3));

	Tuple t;


	//t.cache_lines.assign((byte*) V_host1.data(), (byte*) V_host1.data() + (sizeof(CacheLine<SHARED_LINE_SIZE> ) * V_host1.size()));
	//t.cache_lines2.assign((byte*) V_host2.data(), (byte*) V_host2.data() + (sizeof(CacheLine<SHARED_LINE_SIZE> ) * V_host2.size()));
	//t.cache_lines3.assign((byte*) V_host3.data(), (byte*) V_host3.data() + (sizeof(CacheLine<SHARED_LINE_SIZE> ) * V_host3.size()));
        t.cache_lines = move_to_byte<SHARED_LINE_SIZE>(V_host1);
        t.cache_lines2 = move_to_byte<SHARED_LINE_SIZE>(V_host2);
        t.cache_lines3 = move_to_byte<SHARED_LINE_SIZE>(V_host3);
        
	t.misses = {};

	t.hits = {};
	t.errors = 0;//shared_mem_err_host1;
	t.errors2 = 0;//shared_mem_err_host2;
	t.errors3 = 0;//shared_mem_err_host3;
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
        case XAVIER:
	case TITANV: {
		const uint32 max_shared_mem = 48 * 1024;

		if (max_shared_mem * 2 != parameters.shared_memory_size )
			error(
					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.shared_memory_size));

		const uint32 cache_line_size = 128;
		const uint32 v_size = max_shared_mem / cache_line_size;
		dim3 block_size(parameters.number_of_sms, 4), threads_per_block(v_size);

		return test_shared_memory<v_size, cache_line_size, max_shared_mem>(
				parameters.number_of_sms, parameters.t_byte,
				parameters.one_second_cycles, block_size, threads_per_block);
//		break;
	}
	}

	return Tuple();
}

