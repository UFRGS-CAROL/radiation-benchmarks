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

#include "CacheLine.h"
#include "L2Cache.h"

#include "utils.h"

template<typename int_t, const uint32 V_SIZE, const uint32 LINE_SIZE>
__global__ void test_l2_cache_kernel(CacheLine<LINE_SIZE> *lines1,
		int_t *l2_hit_array, int_t *l2_miss_array, std::int64_t sleep_cycles,
		const uint32 t) {
	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < V_SIZE) {

		register int_t t1 = clock();
		lines1[i] = t;
		register int_t t2 = clock();
		l2_miss_array[i] = t2 - t1;

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//last checking
		t1 = clock();
		CacheLine<LINE_SIZE> r = lines1[i];
		t2 = clock();
		l2_hit_array[i] = t2 - t1;

		lines1[i] = r;
	}
}

__global__ void clear_cache_kenel(float *random_array) {
	register uint32 tx = blockIdx.x * blockDim.x + threadIdx.x;
	random_array[tx] += random_array[tx] * 339 + 1 * (-random_array[tx]);
}

void L2Cache::clear_cache(uint32 n) {
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
	if (thread_number > 1024)
		thread_number = 1024;
	if (block_number > 1024)
		block_number = 1024;

	clear_cache_kenel<<<block_number, thread_number>>>(random_array_dev);
	cuda_check(cudaDeviceSynchronize());

	(curandDestroyGenerator(gen));

	cuda_check(cudaFree(random_array_dev));

}

void L2Cache::test(const uint32& mem) {
	//Clear the L2 Cache
	clear_cache(this->l2_size / sizeof(float));

	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<uint64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<CacheLine<CACHE_LINE_SIZE>> input_device_1 =
			this->input_host_1;

	switch (this->device) {
	case K20:{
		const uint32 max_l2_cache = 1280 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<uint64, v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(), hit_vector_device.data(),
						miss_vector_device.data(), cycles, mem);

		break;
	}

	case K40: {
		const uint32 max_l2_cache = 1536 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<uint64, v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(), hit_vector_device.data(),
						miss_vector_device.data(), cycles, mem);

		break;
	}
	case XAVIER: {
		const uint32 max_l2_cache = 512 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
		  error(
		                  "L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
		                                + std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<uint64, v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(), hit_vector_device.data(),
						miss_vector_device.data(), cycles, mem);

		break;
	}
	case TITANV: {
		const uint32 max_l2_cache = 4608 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<uint64, v_size, CACHE_LINE_SIZE> <<<block_size,
				threads_per_block>>>(input_device_1.data(), hit_vector_device.data(),
						miss_vector_device.data(), cycles, mem);

		break;
	}
	}

	cuda_check(cudaDeviceSynchronize());

	//Host arrays
	//Copy back to the host
	this->hit_vector_host = hit_vector_device.to_vector();
	this->miss_vector_host = miss_vector_device.to_vector();
	this->output_host_1 = input_device_1.to_vector();

}

L2Cache::L2Cache(const Parameters& parameters) : Memory<CacheLine<CACHE_LINE_SIZE> >(parameters){
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	/**
	 * Split alongside the blocks
	 */
	this->threads_per_block = dim3(BLOCK_SIZE);
	uint32 v_size;
	switch (this->device) {
	case K20:{
		const uint32 max_l2_cache = 1280 * 1024;
		if (max_l2_cache != parameters.l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.l2_size));

		v_size = max_l2_cache / CACHE_LINE_SIZE;
		this->block_size = dim3(v_size / BLOCK_SIZE);
		break;
	}
	case K40: {
		const uint32 max_l2_cache = 1536 * 1024; //bytes
		if (max_l2_cache != parameters.l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.l2_size));

		v_size = max_l2_cache / CACHE_LINE_SIZE;
		this->block_size = dim3(v_size / BLOCK_SIZE);
		break;
	}
	case XAVIER: {
		const uint32 max_l2_cache = 512 * 1024; //bytes
		if (max_l2_cache != parameters.l2_size)
		  error(
		                  "L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
		                                + std::to_string(parameters.l2_size));

		v_size = max_l2_cache / CACHE_LINE_SIZE;
		this->block_size = dim3(v_size / BLOCK_SIZE);
		break;
	}
	case TITANV: {
		const uint32 max_l2_cache = 4608 * 1024; //bytes
		if (max_l2_cache != parameters.l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(parameters.l2_size));

		v_size = max_l2_cache / CACHE_LINE_SIZE;
		this->block_size = dim3(v_size / BLOCK_SIZE);
		break;
	}
	}

	this->hit_vector_host.resize(v_size, 0);
	this->miss_vector_host.resize(v_size, 0);
	this->input_host_1.resize(v_size, 0);
	this->l2_size = parameters.l2_size;
}

std::string L2Cache::error_detail(uint32 i, uint32 e, uint32 r, uint64 hits,
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

void L2Cache::call_checker(const std::vector<CacheLine<CACHE_LINE_SIZE>>& v1,
		const uint32& valGold, Log& log, uint64 hits, uint64 misses,
		uint64 false_hits, bool verbose) {

	this->check_output_errors((uint32*) v1.data(), valGold, log, hits, misses,
			false_hits, verbose,
			v1.size() * CHUNK_SIZE(CACHE_LINE_SIZE, uint32));
}
