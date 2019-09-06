/*
 * shared_memory_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#include "CacheLine.h"
#include "utils.h"
#include "SharedMemory.h"


template<const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_shared_memory_kernel(CacheLine<LINE_SIZE> *lines1,
		std::int64_t sleep_cycles, const uint32 t) {

	__shared__ CacheLine<LINE_SIZE> V[V_SIZE];

	if (threadIdx.x < V_SIZE && blockIdx.y == 0) {
		V[threadIdx.x] = t; //lines[blockIdx.x * V_SIZE + threadIdx.x];

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		lines1[blockIdx.x * V_SIZE + threadIdx.x] = V[threadIdx.x];
	}
}


void SharedMemory::test(const uint32& mem) {
//
//	uint32 v_size_multiple_threads = V_SIZE * number_of_sms;
//
//	//Set each element of V array
//	CacheLine<SHARED_LINE_SIZE> *V_dev1, *V_dev2, *V_dev3;
//	cuda_check(
//			cudaMalloc(&V_dev1,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads));
//	cuda_check(
//			cudaMalloc(&V_dev2,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads));
//	cuda_check(
//			cudaMalloc(&V_dev3,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads));
//
//	//Set the number of threads
//	//These archs support two blocks per SM with 48KB of shared memory
//	test_shared_memory_kernel<V_SIZE, SHARED_LINE_SIZE> <<<block_size,
//			threads_per_block>>>(V_dev1, V_dev2, V_dev3, cycles, t_byte);
//	cuda_check(cudaDeviceSynchronize());
//
//	//V array host
//	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host1(v_size_multiple_threads,
//			0);
//	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host2(v_size_multiple_threads,
//			0);
//	std::vector<CacheLine<SHARED_LINE_SIZE>> V_host3(v_size_multiple_threads,
//			0);
//
//	cuda_check(
//			cudaMemcpy(V_host1.data(), V_dev1,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads, cudaMemcpyDeviceToHost));
//	cuda_check(
//			cudaMemcpy(V_host2.data(), V_dev2,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads, cudaMemcpyDeviceToHost));
//	cuda_check(
//			cudaMemcpy(V_host3.data(), V_dev3,
//					sizeof(CacheLine<SHARED_LINE_SIZE> )
//							* v_size_multiple_threads, cudaMemcpyDeviceToHost));


}

/**
 * Shared memory size is in bytes
 */
SharedMemory::SharedMemory(const Parameters& parameters) : Memory<CacheLine<CACHE_LINE_SIZE>> (parameters) {
//This switch is only to set manually the cache line size
//since it is hard to check it at runtime
//	switch (parameters.device) {
//	case K40: {
//		const uint32 max_shared_mem = 48 * 1024;
//
//		if (max_shared_mem != parameters.shared_memory_size)
//			error(
//					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
//							+ std::to_string(parameters.shared_memory_size));
//
//		const uint32 cache_line_size = 128;
//		const uint32 v_size = max_shared_mem / cache_line_size;
//		dim3 block_size(parameters.number_of_sms, 1), threads_per_block(v_size);
//
//		return test_shared_memory<v_size, cache_line_size, max_shared_mem>(
//				parameters.number_of_sms, parameters.t_byte,
//				parameters.one_second_cycles, block_size, threads_per_block);
////		break;
//	}
//	case XAVIER:
//	case TITANV: {
//		const uint32 max_shared_mem = 48 * 1024;
//
//		if (max_shared_mem * 2 != parameters.shared_memory_size)
//			error(
//					"SHARED DEFAULT SIZE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
//							+ std::to_string(parameters.shared_memory_size));
//
//		const uint32 cache_line_size = 128;
//		const uint32 v_size = max_shared_mem / cache_line_size;
//		dim3 block_size(parameters.number_of_sms, 4), threads_per_block(v_size);
//
//		return test_shared_memory<v_size, cache_line_size, max_shared_mem>(
//				parameters.number_of_sms, parameters.t_byte,
//				parameters.one_second_cycles, block_size, threads_per_block);
////		break;
//	}
//	}
}

std::string SharedMemory::error_detail(uint32 i, uint32 e, uint32 r, uint64 hits,
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

void SharedMemory::call_checker(const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
		const uint32& valGold, Log& log, uint64 hits, uint64 misses,
		uint64 false_hits, bool verbose) {

	this->check_output_errors((uint32*) v1.data(), valGold, log, hits, misses,
			false_hits, verbose,
			v1.size() * CHUNK_SIZE(CACHE_LINE_SIZE, uint32));
}
