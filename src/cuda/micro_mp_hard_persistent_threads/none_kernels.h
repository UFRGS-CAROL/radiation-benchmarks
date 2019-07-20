/*
 * none_kernels.h
 *
 *  Created on: 31/03/2019
 *      Author: fernando
 */

#ifndef NONE_KERNELS_H_
#define NONE_KERNELS_H_

#include "include/persistent_lib.h"

template<typename real>
struct Input {
	real OUTPUT_R;
	real INPUT_A;
	real INPUT_B;

	Input(real out, real inp_a, real inp_b) :
			OUTPUT_R(out), INPUT_A(inp_a), INPUT_B(INPUT_B) {
	}

	Input() {
	}

};

template<typename full>
__device__ void process_data(full* d_R0_one, const full OUTPUT_R, const full INPUT_A,
		const full INPUT_B) {
	register volatile full acc_full = OUTPUT_R;
	register volatile full input_a_full = INPUT_A;
	register volatile full input_b_full = INPUT_B;
	register volatile full input_a_neg_full = -INPUT_A;
	register volatile full input_b_neg_full = -INPUT_B;
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
 * FMA
 * ----------------------------------------
 */

template<typename full>
__global__ void MicroBenchmarkKernel_FMA(full *d_R0_one, const full OUTPUT_R,
		const full INPUT_A, const full INPUT_B) {
	rad::PersistentKernel pk;

	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			process_data(d_R0_one, OUTPUT_R, INPUT_A, INPUT_B);
			pk.iteration_finished();
		}
	}

}

template<typename full>
__device__ void process_data(const Input<full>* input, full* d_R0_one) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	const Input<full>& inp_id = input[id];
	register volatile full acc_full = inp_id.OUTPUT_R;
	register volatile full input_a_full = inp_id.INPUT_A;
	register volatile full input_b_full = inp_id.INPUT_B;
	register volatile full input_a_neg_full = -inp_id.INPUT_A;
	register volatile full input_b_neg_full = -inp_id.INPUT_B;
	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = fma_dmr(input_a_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_full, acc_full);
		acc_full = fma_dmr(input_a_full, input_b_neg_full, acc_full);
		acc_full = fma_dmr(input_a_neg_full, input_b_neg_full, acc_full);
	}
	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

template<typename full>
__global__ void MicroBenchmarkKernel_FMA(full *d_R0_one,
		const Input<full>* input) {
	rad::PersistentKernel pk;

	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			process_data(input, d_R0_one);
			pk.iteration_finished();
		}
	}
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
