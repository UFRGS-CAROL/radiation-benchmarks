/*
 * none_kernels.h
 *
 *  Created on: 31/03/2019
 *      Author: fernando
 */

#ifndef NONE_KERNELS_H_
#define NONE_KERNELS_H_

#include "common.h"
#include "device_functions.h"
#include "input_constant.h"

/**
 * ----------------------------------------
 * FMA
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_fma(real_t* output_1, real_t* output_2,
		real_t* output_3) {
	register real_t acc_real_t = 0.0;
	register real_t this_thread_input_real_t = real_t(input_constant_fma[threadIdx.x]);

#pragma unroll
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = fma_dmr(this_thread_input_real_t, this_thread_input_real_t,
				acc_real_t);
	}

	const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;
}

/**
 * ----------------------------------------
 * ADD
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_add(real_t* output_1, real_t* output_2,
		real_t* output_3) {
	register real_t this_thread_input_real_t = real_t(input_constant_add[threadIdx.x]);
	register real_t acc_real_t = 0.0;

#pragma unroll
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = add_dmr(this_thread_input_real_t, acc_real_t);
	}

	const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;
}

/**
 * ----------------------------------------
 * MUL
 * ----------------------------------------
 */

template<typename real_t>
__global__ void microbenchmark_kernel_mul(real_t* output_1, real_t* output_2,
		real_t* output_3) {
	const register real_t this_thread_input_real_t = real_t(input_constant_mul[threadIdx.x]);
	register real_t acc_real_t = this_thread_input_real_t;

#pragma unroll
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = mul_dmr(this_thread_input_real_t, acc_real_t);
	}

	const register uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	output_1[thread_id] = acc_real_t;
	output_2[thread_id] = acc_real_t;
	output_3[thread_id] = acc_real_t;
}

#endif /* NONE_KERNELS_H_ */
