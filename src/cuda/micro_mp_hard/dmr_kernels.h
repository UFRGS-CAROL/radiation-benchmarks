/*
 * dmr_kernels.cu
 *
 *  Created on: 26/03/2019
 *      Author: fernando
 */

#ifndef DMR_KERNELS_CU_
#define DMR_KERNELS_CU_

#include "device_functions.h"

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_FMA(incomplete *d_R0_one,
		full *d_R0_second, full error_threshold, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
	register full acc_full = OUTPUT_R;
	register full input_a_full = INPUT_A;
	register full input_b_full = INPUT_B;
	register full input_a_neg_full = -INPUT_A;
	register full input_b_neg_full = -INPUT_B;

	register incomplete acc_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_incomplete = incomplete(INPUT_A);
	register incomplete input_b_incomplete = incomplete(INPUT_B);
	register incomplete input_a_neg_incomplete = incomplete(-INPUT_A);
	register incomplete input_b_neg_incomplete = incomplete(-INPUT_B);

#pragma unroll 512
	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = fma_dmr(input_a_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_full, input_b_neg_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_neg_full, acc_full);

		acc_incomplete = fma_dmr(input_a_incomplete, input_b_incomplete,
				acc_incomplete);
		acc_incomplete = fma_dmr(input_a_neg_incomplete, input_b_incomplete,
				acc_incomplete);
		acc_incomplete = fma_dmr(input_a_incomplete, input_b_neg_incomplete,
				acc_incomplete);
		acc_incomplete = fma_dmr(input_a_neg_incomplete, input_b_neg_incomplete,
				acc_incomplete);

	}

	check_relative_error(acc_incomplete, acc_full, error_threshold);

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

/**
 * ----------------------------------------
 * ADD DMR
 * ----------------------------------------
 */

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_ADD(incomplete *d_R0_one,
		full *d_R0_second, full error_threshold, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
// ========================================== Double and Single precision
	register full acc_full = OUTPUT_R;
	register full input_a = OUTPUT_R;
	register full input_a_neg = -OUTPUT_R;

	register incomplete acc_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_neg_incomplete = incomplete(-OUTPUT_R);

#pragma unroll 512
	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = add_dmr(acc_full, input_a);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a);

		acc_incomplete = add_dmr(acc_incomplete, input_a_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_neg_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_neg_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_incomplete);
	}

	check_relative_error(acc_incomplete, acc_full, error_threshold);

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

/**
 * ----------------------------------------
 * MUL DMR
 * ----------------------------------------
 */

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_MUL(incomplete *d_R0_one,
		full *d_R0_second, full error_threshold, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {

	register full acc_full = OUTPUT_R;
	register full input_a_full = INPUT_A;
	register full input_a_inv_full = full(1.0) / INPUT_A;

	register incomplete acc_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_incomplete = incomplete(INPUT_B);
	register incomplete input_a_inv_incomplete = incomplete(1.0) / incomplete(INPUT_B);

#pragma unroll 512
	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = mul_dmr(acc_full, input_a_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_full);

		acc_incomplete = mul_dmr(acc_incomplete, input_a_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_inv_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_inv_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_incomplete);
	}

	check_relative_error(acc_incomplete, acc_full, error_threshold);

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

#endif /* DMR_KERNELS_CU_ */
