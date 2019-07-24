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

	double theshold = -2222;
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
		theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
	theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	}

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
		full *d_R0_second, const full OUTPUT_R, const full INPUT_A) {
	// ========================================== Double and Single precision
	register full acc_full = OUTPUT_R;
	register full input_a = OUTPUT_R;
	register full input_a_neg = -OUTPUT_R;

	register incomplete acc_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_incomplete = incomplete(OUTPUT_R);
	register incomplete input_a_neg_incomplete = incomplete(-OUTPUT_R);
	double theshold = -2222;

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
		theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
	theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	}

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
		full *d_R0_second, const full OUTPUT_R, const full INPUT_A) {

	register full acc_full = OUTPUT_R;
	register full input_a_full = INPUT_A;
	register full input_a_inv_full = full(1.0) / INPUT_A;

	register incomplete acc_incomplete = incomplete(acc_full);
	register incomplete input_a_incomplete = incomplete(input_a_full);
	register incomplete input_a_inv_incomplete = incomplete(input_a_inv_full);
	double theshold = -2222;

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
		theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

		acc_incomplete = incomplete(acc_full);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
	theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	}
	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;
}

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_ADDNOTBIASAED(incomplete *d_R0_one,
		full *d_R0_second, const full OUTPUT_R) {
	register full divisor = full(NUM_COMPOSE_DIVISOR);
	register full acc_full = 0.0;
	register full slice_full = OUTPUT_R / divisor;

	register incomplete acc_incomplete = 0.0;
	register incomplete slice_incomplete = incomplete(slice_full);
//	double theshold = -2222;

	for (int count = 0; count < NUM_COMPOSE_DIVISOR; count++) {
		acc_full = add_dmr(slice_full, acc_full);

		acc_incomplete = add_dmr(slice_incomplete, acc_incomplete);

#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
//		theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

		acc_incomplete = incomplete(acc_full);
// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);
			//theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
	//theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

//	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
//	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_MULNOTBIASAED(incomplete *d_R0_one,
		full *d_R0_second, const full OUTPUT_R) {
	register full acc_full = OUTPUT_R / full(DIV_FOR_MUL);
	register incomplete acc_incomplete = incomplete(OUTPUT_R)
			/ incomplete(DIV_FOR_MUL);
//	double theshold = -2222;

	const register full f = acc_full;
	const register incomplete i = acc_incomplete;

	for (int count = 0; count < NUM_MUL_OP; count++) {
		acc_full = mul_dmr(acc_full, f);
		acc_incomplete = mul_dmr(acc_incomplete, i);

#if CHECKBLOCK == 1
		check_relative_error(acc_incomplete, acc_full);
//		theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

		acc_incomplete = incomplete(acc_full);
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);

//			theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
//	theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

//	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
//	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_FMANOTBIASAED(incomplete *d_R0_one,
		full *d_R0_second, const full OUTPUT_R) {
	register full acc_full = OUTPUT_R / full(DIV_FOR_MUL);
	register incomplete acc_incomplete = incomplete(OUTPUT_R) / incomplete(DIV_FOR_MUL);

	register full b_full = acc_full / full(NUM_MUL_OP);
	register incomplete b_incomplete = acc_incomplete / incomplete(NUM_MUL_OP);


	const register full a_full = acc_full;
	const register incomplete a_incomplete = acc_incomplete;

	double theshold = -2222;

	for (int count = 0; count < NUM_MUL_OP; count++) {
		acc_full = fma_dmr(a_full, b_full, acc_full);
		acc_incomplete = fma_dmr(a_incomplete, b_incomplete, acc_incomplete);

#if CHECKBLOCK >= 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_incomplete, acc_full);

			theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

			acc_incomplete = incomplete(acc_full);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_incomplete, acc_full);
	theshold = fmax(theshold, fabs(double(acc_full) - double(acc_incomplete)));

#endif

	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

#endif /* DMR_KERNELS_CU_ */
