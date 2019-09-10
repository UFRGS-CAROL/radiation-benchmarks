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

template<const uint32 V_SIZE>
__global__ void test_l2_cache_kernel(uint64 *in, uint64* out,
		int64 *l2_hit_array, int64 *l2_miss_array, int64 sleep_cycles) {
	const register uint32 i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < V_SIZE) {
		register uint64 rs[CACHE_LINE_SIZE_BY_INT32];
		register uint64 rt[CACHE_LINE_SIZE_BY_INT32];

		register const int64 t1_miss = clock64();
		move_cache_line(rs, in + i);
		register const int64 t2_miss = clock64();

		//wait for exposition to neutrons
		sleep_cuda(sleep_cycles);

		//last checking
		register const int64 t1_hit = clock64();
		move_cache_line(rt, in + i);
		register const int64 t2_hit = clock64();

		move_cache_line(in + i, rs);
		move_cache_line(out + i, rt);

		l2_hit_array[i] = t2_hit - t1_hit;
		l2_miss_array[i] = t2_miss - t1_miss;
	}
}

void L2Cache::test(const uint64& mem) {
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<int64> hit_vector_device(this->hit_vector_host);
	rad::DeviceVector<int64> miss_vector_device(this->miss_vector_host);

	rad::DeviceVector<uint64> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint64> output_device_1 = this->input_host_1;

	switch (this->device) {
	case K20: {
		const uint32 max_l2_cache = 1280 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles);
		break;
	}

	case K40: {
		const uint32 max_l2_cache = 1536 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles);
		break;
	}
	case XAVIER: {
		const uint32 max_l2_cache = 512 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles);
		break;
	}
	case TITANV: {
		const uint32 max_l2_cache = 4608 * 1024; //bytes
		if (max_l2_cache != this->l2_size)
			error(
					"L2 DEFAULT CACHE AND DRIVER OBTAINED VALUE DOES NOT MACH. REAL VALUE:"
							+ std::to_string(this->l2_size));

		constexpr uint32 v_size = max_l2_cache / CACHE_LINE_SIZE;
		test_l2_cache_kernel<v_size> <<<block_size,
				threads_per_block>>>(input_device_1.data(),
				output_device_1.data(), hit_vector_device.data(),
				miss_vector_device.data(), cycles);
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

L2Cache::L2Cache(const Parameters& parameters) :
		Memory<uint64>(parameters) {
	//This switch is only to set manually the cache line size
	//since it is hard to check it at runtime
	/**
	 * Split alongside the blocks
	 */
	this->threads_per_block = dim3(BLOCK_SIZE);
	uint32 v_size;
	switch (this->device) {
	case K20: {
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
