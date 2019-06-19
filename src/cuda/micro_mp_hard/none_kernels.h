/*
 * none_kernels.h
 *
 *  Created on: 31/03/2019
 *      Author: fernando
 */

#ifndef NONE_KERNELS_H_
#define NONE_KERNELS_H_

#include "device_functions.h"

/**
 * ----------------------------------------
 * FMA
 * ----------------------------------------
 */

template<typename full>
__global__ void MicroBenchmarkKernel_FMA(full *d_R0_one, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a_full = INPUT_A;
	volatile register full input_b_full = INPUT_B;
	volatile register full input_a_neg_full = -INPUT_A;
	volatile register full input_b_neg_full = -INPUT_B;

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = fma_dmr(input_a_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_full, input_b_neg_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_neg_full, acc_full);

	}
	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

/**
 * ----------------------------------------
 * ADD
 * ----------------------------------------
 */

template<typename full>
__global__ void MicroBenchmarkKernel_ADD(full *d_R0_one, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a = OUTPUT_R;
	volatile register full input_a_neg = -OUTPUT_R;

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = add_dmr(acc_full, input_a);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a);
	}
	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

/**
 * ----------------------------------------
 * MUL
 * ----------------------------------------
 */

template<typename full>
__global__ void MicroBenchmarkKernel_MUL(full *d_R0_one, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a_full = INPUT_A;
	volatile register full input_a_inv_full = full(1.0) / INPUT_A;

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = mul_dmr(acc_full, input_a_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_full);
	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

#endif /* NONE_KERNELS_H_ */
