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
		full *d_R0_second, const full OUTPUT_R, const full INPUT_A,
		const full INPUT_B) {
	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a_full = INPUT_A;
	volatile register full input_b_full = INPUT_B;
	volatile register full input_a_neg_full = -INPUT_A;
	volatile register full input_b_neg_full = -INPUT_B;

	volatile register incomplete acc_incomplete = incomplete(OUTPUT_R);
	volatile register incomplete input_a_incomplete = incomplete(INPUT_A);
	volatile register incomplete input_b_incomplete = incomplete(INPUT_B);
	volatile register incomplete input_a_neg_incomplete = incomplete(-INPUT_A);
	volatile register incomplete input_b_neg_incomplete = incomplete(-INPUT_B);

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

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
#endif

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
		full *d_R0_second, const full OUTPUT_R, const full INPUT_A,
		const full INPUT_B) {
	// ========================================== Double and Single precision
	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a = OUTPUT_R;
	volatile register full input_a_neg = -OUTPUT_R;

	volatile register incomplete acc_incomplete = incomplete(OUTPUT_R);
	volatile register incomplete input_a_incomplete = incomplete(OUTPUT_R);
	volatile register incomplete input_a_neg_incomplete = incomplete(-OUTPUT_R);

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = add_dmr(acc_full, input_a);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a_neg);
		acc_full = add_dmr(acc_full, input_a);

		acc_incomplete = add_dmr(acc_incomplete, input_a_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_neg_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_neg_incomplete);
		acc_incomplete = add_dmr(acc_incomplete, input_a_incomplete);

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			acc_incomplete = incomplete(acc_full);
		}
#endif
	}

	//if CHECKBLOCK is 0 only after OPS iterations it will be verified
#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
#endif

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
		full *d_R0_second, const full OUTPUT_R, const full INPUT_A,
		const full INPUT_B) {

	volatile register full acc_full = OUTPUT_R;
	volatile register full input_a_full = INPUT_A;
	volatile register full input_a_inv_full = full(1.0) / INPUT_A;

	volatile register incomplete acc_incomplete = incomplete(OUTPUT_R);
	volatile register incomplete input_a_incomplete = incomplete(INPUT_A);
	volatile register incomplete input_a_inv_incomplete = incomplete(1.0)
			/ incomplete(INPUT_A);

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_full = mul_dmr(acc_full, input_a_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_inv_full);
		acc_full = mul_dmr(acc_full, input_a_full);

		acc_incomplete = mul_dmr(acc_incomplete, input_a_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_inv_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_inv_incomplete);
		acc_incomplete = mul_dmr(acc_incomplete, input_a_incomplete);

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

	//if CHECKBLOCK is 0 only after OPS iterations it will be verified
#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
#endif

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_SQRT(incomplete *d_R0_one,
		full *d_R0_second, const full INPUT_A) {

}

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_NumCompose(incomplete *d_R0_one,
		full *d_R0_second, const full OUTPUT_R) {
	int n = 10000000;
	const full divisor = full(n);
	volatile full acc_full = 0.0;
	volatile full slice_full = OUTPUT_R / divisor;

	volatile full acc_incomplete = 0.0;
	volatile full slice_incomplete = incomplete(OUTPUT_R) / incomplete(divisor);

	for (int i = 0; i < n; i++) {
		acc_full = add_dmr(slice_full, acc_full);

		acc_incomplete = add_dmr(slice_incomplete, acc_incomplete);

#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((i % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

	//if CHECKBLOCK is 0 only after OPS iterations it will be verified
#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
#endif

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

#endif /* DMR_KERNELS_CU_ */
