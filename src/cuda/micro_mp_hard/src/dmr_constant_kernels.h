/*
 * dmr_nonconstant_kernels.h
 *
 *  Created on: Jul 30, 2019
 *      Author: fernando
 */

#ifndef DMR_CONSTANT_KERNELS_H_
#define DMR_CONSTANT_KERNELS_H_

#include "common.h"
#include "device_functions.h"
#include "input_constant.h"

template<typename real_t>
__global__ void compare(real_t* lhs, real_t* rhs) {
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	check_bit_error<0>(lhs[tid], rhs[tid]);
}

template<const uint32 THRESHOLD, const uint32 COUNT, typename real_t,
		typename half_t>
__global__ void microbenchmark_kernel_add(real_t* output_real_t_1,
		real_t* output_real_t_2, real_t* output_real_t_3,
		half_t* output_half_t_1, half_t* output_half_t_2,
		half_t* output_half_t_3) {

	register real_t acc_real_t = 0.0;
	register half_t acc_half_t = 0.0;

	const register real_t this_thread_input_real_t = real_t(
			input_constant_add[threadIdx.x]);
	const register half_t this_thread_input_half_t = half_t(
			this_thread_input_real_t);
//#pragma unroll COUNT
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = add_dmr(this_thread_input_real_t, acc_real_t);
		acc_half_t = add_dmr(this_thread_input_half_t, acc_half_t);

		if ((count % COUNT) == 0) {
			check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
			acc_half_t = half_t(acc_real_t);
		}
	}

	if (COUNT == OPS) {
		check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
	}

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	output_real_t_1[thread_id] = acc_real_t;
	output_real_t_2[thread_id] = acc_real_t;
	output_real_t_3[thread_id] = acc_real_t;

	output_half_t_1[thread_id] = acc_half_t;
	output_half_t_2[thread_id] = acc_half_t;
	output_half_t_3[thread_id] = acc_half_t;
}

template<const uint32 THRESHOLD, const uint32 COUNT, typename real_t,
		typename half_t>
__global__ void microbenchmark_kernel_mul(real_t* output_real_t_1,
		real_t* output_real_t_2, real_t* output_real_t_3,
		half_t* output_half_t_1, half_t* output_half_t_2,
		half_t* output_half_t_3) {

	const register real_t this_thread_input_real_t = real_t(
			input_constant_mul[threadIdx.x]);
	const register half_t this_thread_input_half_t = half_t(
			this_thread_input_real_t);

	register real_t acc_real_t = this_thread_input_real_t;
	register half_t acc_half_t = this_thread_input_half_t;

//#pragma unroll COUNT
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = mul_dmr(this_thread_input_real_t, acc_real_t);
		acc_half_t = mul_dmr(this_thread_input_half_t, acc_half_t);

		if ((count % COUNT) == 0) {
			check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
			acc_half_t = half_t(acc_real_t);
		}
	}

	if (COUNT == OPS) {
		check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
	}

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	output_real_t_1[thread_id] = acc_real_t;
	output_real_t_2[thread_id] = acc_real_t;
	output_real_t_3[thread_id] = acc_real_t;

	output_half_t_1[thread_id] = acc_half_t;
	output_half_t_2[thread_id] = acc_half_t;
	output_half_t_3[thread_id] = acc_half_t;
}

template<const uint32 THRESHOLD, const uint32 COUNT, typename real_t,
		typename half_t>
__global__ void microbenchmark_kernel_fma(real_t* output_real_t_1,
		real_t* output_real_t_2, real_t* output_real_t_3,
		half_t* output_half_t_1, half_t* output_half_t_2,
		half_t* output_half_t_3) {

	register real_t acc_real_t = 0.0;
	register half_t acc_half_t = 0.0;

	const register real_t this_thread_input_real_t = real_t(
			input_constant_fma[threadIdx.x]);
	const register half_t this_thread_input_half_t = half_t(
			this_thread_input_real_t);

//#pragma unroll COUNT
	for (uint32 count = 0; count < OPS; count++) {
		acc_real_t = fma_dmr(this_thread_input_real_t, this_thread_input_real_t,
				acc_real_t);
		acc_half_t = fma_dmr(this_thread_input_half_t, this_thread_input_half_t,
				acc_half_t);

		if ((count % COUNT) == 0) {
			check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
			acc_half_t = half_t(acc_real_t);
		}
	}

	if (COUNT == OPS) {
		check_bit_error<THRESHOLD, COUNT>(acc_half_t, acc_real_t);
	}

	const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	output_real_t_1[thread_id] = acc_real_t;
	output_real_t_2[thread_id] = acc_real_t;
	output_real_t_3[thread_id] = acc_real_t;

	output_half_t_1[thread_id] = acc_half_t;
	output_half_t_2[thread_id] = acc_half_t;
	output_half_t_3[thread_id] = acc_half_t;
}

#endif /* DMR_CONSTANT_KERNELS_H_ */
