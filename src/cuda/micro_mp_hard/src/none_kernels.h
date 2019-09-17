/*
 * none_kernels.h
 *
 *  Created on: 31/03/2019
 *      Author: fernando
 */

#ifndef NONE_KERNELS_H_
#define NONE_KERNELS_H_

#include "device_functions.h"
#include "common.h"
#include "input_constant.h"

/**
 * ----------------------------------------
 * FMA
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_fma(real_t* output_1, real_t* output_2,
		real_t* output_3) {

	__shared__ real_t shared_mem[MAXSHAREDMEMORY / sizeof(real_t)];
	shared_mem[threadIdx.x] = 0;


	const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	register real_t acc_real_t = 0.0;
	register real_t this_thread_input_real_t = real_t(
			input_constant[threadIdx.x]);

#pragma unroll
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = fma_dmr(this_thread_input_real_t, this_thread_input_real_t,
				acc_real_t);
	}

	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;
	acc_real_t = shared_mem[threadIdx.x];
}

/**
 * ----------------------------------------
 * ADD
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_add(real_t* output_1, real_t* output_2,
		real_t* output_3) {
	__shared__ real_t shared_mem[MAXSHAREDMEMORY / sizeof(real_t)];
	shared_mem[threadIdx.x] = 0;

	const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	register real_t this_thread_input_real_t = real_t(
			input_constant[threadIdx.x]);
	register real_t acc_real_t = 0.0;

#pragma unroll
	for (int count = 0; count < OPS; count++) {
		acc_real_t = add_dmr(this_thread_input_real_t, acc_real_t);
	}

	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;
	acc_real_t = shared_mem[threadIdx.x];
}

/**
 * ----------------------------------------
 * MUL
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_mul(real_t* output_1, real_t* output_2,
		real_t* output_3) {
	__shared__ real_t shared_mem[MAXSHAREDMEMORY / sizeof(real_t)];
	shared_mem[threadIdx.x] = 0;

	const register uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	const register real_t this_thread_input_real_t = real_t(input_constant[threadIdx.x]);
	register real_t acc_real_t = this_thread_input_real_t;

#pragma unroll
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = mul_dmr(this_thread_input_real_t, acc_real_t);
	}

//	if(acc_real_t != 0.0)
//		printf("%f %f %d\n", this_thread_input_real_t, acc_real_t, OPS);

	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;

	acc_real_t = shared_mem[threadIdx.x];
}

#endif /* NONE_KERNELS_H_ */
