/*
 * read_only_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "kernels.h"
#include "utils.h"
#include <cstring>

__device__ uint64 read_only_cache_errors;

template<uint32 READ_ONLY_MEM_SIZE, uint32 SHARED_PER_SM>
__global__ void test_read_only_kernel(
		const uint32 * __restrict__ constant_mem_array, uint32 *output_array, int64 sleep_cycles, uint32 t_int) {

	__shared__ uint32 max_shared_mem[SHARED_PER_SM];
	uint32 tx = blockIdx.x * READ_ONLY_MEM_SIZE + threadIdx.x + threadIdx.y;


	if (tx < READ_ONLY_MEM_SIZE) {

		//ldg is a direct load to const memory
		//first round
		const volatile uint32 first_round = __ldg(&constant_mem_array[tx]);
		max_shared_mem[threadIdx.x] = first_round;

		sleep_cuda(sleep_cycles);

		if (t_int != first_round) {
			atomicAdd(&read_only_cache_errors, 1);
		}

		output_array[tx] = first_round;
		max_shared_mem[tx] += first_round;
	}
	__syncthreads();
}

template<uint32 READ_ONLY_MEM_SIZE, uint32 SHARED_PER_SM>
Tuple test_read_only_cache(const uint32 number_of_sms, const uint32 t_int,
		const int64 cycles, dim3& block_size, dim3& threads_per_block) {

	const uint32 v_size_multiple_threads = READ_ONLY_MEM_SIZE * number_of_sms; // Each block with one thread using all l1 cache

	//device arrays
	//Set each element of V array
	uint32 *V_dev, *V_out;
	std::vector<uint32> V_host(v_size_multiple_threads, t_int);

	//copy to the GPU
	cuda_check(cudaMalloc(&V_dev, sizeof(uint32) * v_size_multiple_threads));
	cuda_check(cudaMemcpy(V_dev, V_host.data(),	sizeof(uint32) * v_size_multiple_threads,
					cudaMemcpyHostToDevice));

	//alloc the output vector
	cuda_check(cudaMalloc(&V_out, sizeof(uint32) * v_size_multiple_threads));


	//Set to zero err_check
	uint64 read_only_cache_errors_host = 0;
	cuda_check(
			cudaMemcpyToSymbol(read_only_cache_errors, &read_only_cache_errors_host, sizeof(uint64),
					0));

//	dim3 block_size(number_of_sms, number_of_sms), threads_per_block(V_SIZE);

	test_read_only_kernel<READ_ONLY_MEM_SIZE, SHARED_PER_SM> <<<
			block_size, threads_per_block>>>(V_dev, V_out,  cycles, t_int);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(
			cudaMemcpy(V_host.data(), V_out,
					sizeof(uint32) * v_size_multiple_threads,
					cudaMemcpyDeviceToHost));

	cuda_check(
			cudaMemcpyFromSymbol(&read_only_cache_errors_host, read_only_cache_errors,
					sizeof(uint64), 0));

	cuda_check(cudaFree(V_dev));
	cuda_check(cudaFree(V_out));


	Tuple t;

	t.register_file = std::move(V_host);
	t.errors = read_only_cache_errors_host;

	return t;
}

Tuple test_read_only_cache(const Parameters& parameters) {
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	 uint32 const_data;
		std::memset(&const_data, parameters.t_byte, sizeof(uint32));

	switch (parameters.device) {
	case K20:
	case K40: {
		const uint32 max_constant_cache = 32 * 1024 / sizeof(uint32); //bytes
		const uint32 max_shared_mem = 8 * 1024;
		const uint32 v_size = max_constant_cache;

		if (max_constant_cache * sizeof(uint32) != parameters.const_memory_per_block)
					error(
							"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
									+ std::to_string(parameters.const_memory_per_block));

		dim3 block_size(parameters.number_of_sms), threads_per_block(BLOCK_SIZE * BLOCK_SIZE, v_size / (BLOCK_SIZE * BLOCK_SIZE));

		return test_read_only_cache<v_size, max_shared_mem>(
				parameters.number_of_sms, const_data, parameters.one_second_cycles,
				block_size, threads_per_block);
//		break;
	}
	case TITANV: {
		const uint32 max_constant_cache = 64 * 1024 / sizeof(uint32); //bytes
		const uint32 max_shared_mem = 8 * 1024;
		const uint32 v_size = max_constant_cache;

		if (max_constant_cache * sizeof(uint32) != parameters.const_memory_per_block)
					error(
							"CONST DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
									+ std::to_string(parameters.const_memory_per_block));

		//For Maxwell and above each SM can execute 4 blocks
		dim3 block_size(parameters.number_of_sms), threads_per_block(BLOCK_SIZE * BLOCK_SIZE, v_size / (BLOCK_SIZE * BLOCK_SIZE));
		std::cout << block_size.x << std::endl;
		std::cout << threads_per_block.x << " " << threads_per_block.y <<  std::endl;
		return test_read_only_cache<v_size, max_shared_mem>(
				parameters.number_of_sms, const_data, parameters.one_second_cycles,
				block_size, threads_per_block);
//		break;
	}
	}

	return Tuple();
}

