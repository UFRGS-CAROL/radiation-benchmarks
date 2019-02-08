/*
 * l2_cache_kernels.cu
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */
#include <iostream>
#include <vector>
#include <curand.h>
#include <cstdlib>

#include "kernels.h"
#include "CacheLine.h"
#include "utils.h"

__device__ uint64 l2_cache_err;

template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_l2_cache_kernel(CacheLine<LINE_SIZE> *lines,
		int_t *l2_hit_array, int_t *l2_miss_array, std::int64_t sleep_cycles,
		byte t) {
	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < V_SIZE) {

		volatile int_t t1 = clock();
		CacheLine < LINE_SIZE > r = lines[i];
		volatile int_t t2 = clock();
		l2_miss_array[i] = t2 - t1;

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//last checking
		t1 = clock();
		CacheLine < LINE_SIZE > r2 = lines[i];
		t2 = clock();
		l2_hit_array[i] = t2 - t1;

		//bitwise operation
		if (r != t)
			atomicAdd((unsigned long long*) &l2_cache_err, 1);

		lines[i] = r2;
	}
	__syncthreads();
}

__global__ void clear_cache_kenel(float *random_array) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	random_array[tx] += random_array[tx] * 339 + 1 * (-random_array[tx]);
}

void clear_cache(uint32 n) {
	float *random_array_dev;
	/* Allocate n floats on device */
	cuda_check(cudaMalloc((void ** )&random_array_dev, n * sizeof(float)));

	/* Create pseudo-random number generator */
	curandGenerator_t gen;

	(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	/* Set seed */
	(curandSetPseudoRandomGeneratorSeed(gen, std::rand()));

	/* Generate n floats on device */
	(curandGenerateUniform(gen, random_array_dev, n));

	uint32 thread_number = std::ceil(float(n) / (BLOCK_SIZE * BLOCK_SIZE));
	uint32 block_number = std::ceil(n / float(thread_number));

	clear_cache_kenel<<<block_number, thread_number>>>(random_array_dev);
	cuda_check(cudaDeviceSynchronize());

	(curandDestroyGenerator(gen));

	cuda_check(cudaFree(random_array_dev));

}

template<const uint32 V_SIZE, const uint32 L2_LINE_SIZE>
Tuple test_l2_cache(const byte t_byte, const int64 cycles,
		const uint32 l2_size) {
	//device arrays
	int32 *l2_hit_array_device, *l2_miss_array_device;
	cudaMalloc(&l2_hit_array_device, sizeof(int32) * V_SIZE);
	cudaMalloc(&l2_miss_array_device, sizeof(int32) * V_SIZE);

	//Host arrays
	std::vector<int32> l2_hit_array_host(V_SIZE), l2_miss_array_host(V_SIZE);

	//Set each element of V array
	CacheLine < L2_LINE_SIZE > *V_dev;
	std::vector<CacheLine<L2_LINE_SIZE> > V_host(V_SIZE, t_byte);

	//copy to the gpu
	cuda_check(cudaMalloc(&V_dev, sizeof(CacheLine<L2_LINE_SIZE> ) * V_SIZE));
	cuda_check(
			cudaMemcpy(V_dev, V_host.data(),
					sizeof(CacheLine<L2_LINE_SIZE> ) * V_SIZE,
					cudaMemcpyHostToDevice));

	//Clear the L2 Cache
	clear_cache(l2_size / sizeof(float));

	/**
	 * Split alongside the blocks
	 */
	dim3 block_size(V_SIZE / (BLOCK_SIZE * BLOCK_SIZE));
	dim3 threads_per_block(BLOCK_SIZE * BLOCK_SIZE);

	test_l2_cache_kernel<int32, V_SIZE, L2_LINE_SIZE> <<<block_size, threads_per_block>>>(V_dev,
			l2_hit_array_device, l2_miss_array_device, cycles, t_byte);
	cuda_check(cudaDeviceSynchronize());

	cuda_check(
			cudaMemcpy(l2_hit_array_host.data(), l2_hit_array_device,
					sizeof(int32) * V_SIZE, cudaMemcpyDeviceToHost));
	cuda_check(
			cudaMemcpy(l2_miss_array_host.data(), l2_miss_array_device,
					sizeof(int32) * V_SIZE, cudaMemcpyDeviceToHost));

	cuda_check(
			cudaMemcpy(V_host.data(), V_dev,
					sizeof(CacheLine<L2_LINE_SIZE> ) * V_SIZE,
					cudaMemcpyDeviceToHost));

	//Set to zero err_check
	uint64 l2_cache_err_host = 0;
	cuda_check(
			cudaMemcpyFromSymbol(&l2_cache_err_host, l2_cache_err,
					sizeof(uint64), 0));

	cuda_check(cudaFree(l2_hit_array_device));
	cuda_check(cudaFree(l2_miss_array_device));
	cuda_check(cudaFree(V_dev));

	Tuple t;

	t.cache_lines.assign((byte*) V_host.data(),
			(byte*) V_host.data()
					+ (sizeof(CacheLine<L2_LINE_SIZE> ) * V_host.size()));
	t.misses = std::move(l2_miss_array_host);

	t.hits = std::move(l2_hit_array_host);
	t.errors = l2_cache_err_host;

	return t;
}

Tuple test_l2_cache(const Parameters& parameters) {
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	switch (parameters.device) {
	case K40: {
		const uint32 max_l2_cache = 1536 * 1024; //bytes
		if (max_l2_cache != parameters.l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.l2_size));

		const uint32 cache_line_size = 32;
		const uint32 v_size = max_l2_cache / cache_line_size;
		return test_l2_cache<v_size, cache_line_size>(parameters.t_byte,
				parameters.one_second_cycles, max_l2_cache);
//		break;
	}
	case TITANV: {
		const uint32 max_l2_cache = 6144 * 1024; //bytes
		if (max_l2_cache != parameters.l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.l2_size));

		const uint32 cache_line_size = 64;
		const uint32 v_size = max_l2_cache / cache_line_size;

		return test_l2_cache<v_size, cache_line_size>(parameters.t_byte,
				parameters.one_second_cycles, max_l2_cache);
//		break;
	}
	}

	return Tuple();
}
