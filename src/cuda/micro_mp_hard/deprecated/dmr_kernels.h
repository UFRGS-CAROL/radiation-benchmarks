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

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_FMA(half_t *d_R0_one,
		real_t *d_R0_second, const real_t OUTPUT_R, const real_t INPUT_A,
		const real_t INPUT_B) {
	register real_t acc_real_t = OUTPUT_R;
	register real_t input_a_real_t = INPUT_A;
	register real_t input_b_real_t = INPUT_B;
	register real_t input_a_neg_real_t = -INPUT_A;
	register real_t input_b_neg_real_t = -INPUT_B;

	register half_t acc_half_t = half_t(OUTPUT_R);
	register half_t input_a_half_t = half_t(INPUT_A);
	register half_t input_b_half_t = half_t(INPUT_B);
	register half_t input_a_neg_half_t = half_t(-INPUT_A);
	register half_t input_b_neg_half_t = half_t(-INPUT_B);

	//double theshold = -2222;
	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_real_t = fma_dmr(input_a_real_t, input_b_real_t, acc_real_t);
		acc_real_t = fma_dmr(input_a_neg_real_t, input_b_real_t, acc_real_t);
		acc_real_t = fma_dmr(input_a_real_t, input_b_neg_real_t, acc_real_t);
		acc_real_t = fma_dmr(input_a_neg_real_t, input_b_neg_real_t, acc_real_t);

		acc_half_t = fma_dmr(input_a_half_t, input_b_half_t,
				acc_half_t);
		acc_half_t = fma_dmr(input_a_neg_half_t, input_b_half_t,
				acc_half_t);
		acc_half_t = fma_dmr(input_a_half_t, input_b_neg_half_t,
				acc_half_t);
		acc_half_t = fma_dmr(input_a_neg_half_t, input_b_neg_half_t,
				acc_half_t);

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_half_t, acc_real_t);
		//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

		acc_half_t = half_t(acc_real_t);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_half_t, acc_real_t);
			//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
//	theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

//	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
//	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;

}

/**
 * ----------------------------------------
 * ADD DMR
 * ----------------------------------------
 */

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_ADD(half_t *d_R0_one,
		real_t *d_R0_second, const real_t OUTPUT_R, const real_t INPUT_A) {
	// ========================================== Double and Single precision
	register real_t acc_real_t = OUTPUT_R;
	register real_t input_a = OUTPUT_R;
	register real_t input_a_neg = -OUTPUT_R;

	register half_t acc_half_t = half_t(OUTPUT_R);
	register half_t input_a_half_t = half_t(OUTPUT_R);
	register half_t input_a_neg_half_t = half_t(-OUTPUT_R);
	//double theshold = -2222;

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_real_t = add_dmr(acc_real_t, input_a);
		acc_real_t = add_dmr(acc_real_t, input_a_neg);
		acc_real_t = add_dmr(acc_real_t, input_a_neg);
		acc_real_t = add_dmr(acc_real_t, input_a);

		acc_half_t = add_dmr(acc_half_t, input_a_half_t);
		acc_half_t = add_dmr(acc_half_t, input_a_neg_half_t);
		acc_half_t = add_dmr(acc_half_t, input_a_neg_half_t);
		acc_half_t = add_dmr(acc_half_t, input_a_half_t);

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_half_t, acc_real_t);
		//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

		acc_half_t = half_t(acc_real_t);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_half_t, acc_real_t);
			//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
	//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

	//if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
	//	printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	//}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;
}

/**
 * ----------------------------------------
 * MUL DMR
 * ----------------------------------------
 */

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_MUL(half_t *d_R0_one,
		real_t *d_R0_second, const real_t OUTPUT_R, const real_t INPUT_A) {

	register real_t acc_real_t = OUTPUT_R;
	register real_t input_a_real_t = INPUT_A;
	register real_t input_a_inv_real_t = real_t(1.0) / INPUT_A;

	register half_t acc_half_t = half_t(acc_real_t);
	register half_t input_a_half_t = half_t(input_a_real_t);
	register half_t input_a_inv_half_t = half_t(input_a_inv_real_t);
	//double theshold = -2222;

	for (register unsigned int count = 0; count < (OPS / 4); count++) {
		acc_real_t = mul_dmr(acc_real_t, input_a_real_t);
		acc_real_t = mul_dmr(acc_real_t, input_a_inv_real_t);
		acc_real_t = mul_dmr(acc_real_t, input_a_inv_real_t);
		acc_real_t = mul_dmr(acc_real_t, input_a_real_t);

		acc_half_t = mul_dmr(acc_half_t, input_a_half_t);
		acc_half_t = mul_dmr(acc_half_t, input_a_inv_half_t);
		acc_half_t = mul_dmr(acc_half_t, input_a_inv_half_t);
		acc_half_t = mul_dmr(acc_half_t, input_a_half_t);

		// if CHECKBLOCK is 1 each iteration will be verified
#if CHECKBLOCK == 1
		check_relative_error(acc_half_t, acc_real_t);
		//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

		acc_half_t = half_t(acc_real_t);
		// if CHECKBLOCK is >1 perform the % operation
#elif CHECKBLOCK > 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_half_t, acc_real_t);
			//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
	//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

	//if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
	//	printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	//}
	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;
}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_ADDNOTBIASAED(half_t *d_R0_one,
		real_t *d_R0_second, const real_t OUTPUT_R) {
	register real_t acc_real_t = 0.0;
	register real_t slice_real_t = real_t(OUTPUT_R) / real_t(NUM_COMPOSE_DIVISOR);

	register half_t acc_half_t = 0.0;
	register half_t slice_half_t = half_t(OUTPUT_R)
			/ half_t(NUM_COMPOSE_DIVISOR);
//	double theshold = -2222;

	for (int count = 0; count < NUM_COMPOSE_DIVISOR; count++) {
		acc_real_t = add_dmr(slice_real_t, acc_real_t);

		acc_half_t = add_dmr(slice_half_t, acc_half_t);

#if CHECKBLOCK >= 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_half_t, acc_real_t);
			//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
	//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

//	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
//	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;

}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_MULNOTBIASAED(half_t *d_R0_one,
		real_t *d_R0_second) {
	register real_t acc_real_t = real_t(MUL_INPUT);
	register half_t acc_half_t = half_t(MUL_INPUT);
	//double theshold = -2222;

	const register real_t f = acc_real_t;
	const register half_t i = acc_half_t;

	for (int count = 0; count < NUM_COMPOSE_DIVISOR; count++) {
		acc_real_t = mul_dmr(acc_real_t, f);
		acc_half_t = mul_dmr(acc_half_t, i);

#if CHECKBLOCK >= 1
		if((count % CHECKBLOCK) == 0) {
			check_relative_error(acc_half_t, acc_real_t);

			//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
	//theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

	//if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
	//	printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	//}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;

}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_FMANOTBIASAED(half_t *d_R0_one,
		real_t *d_R0_second) {
	register real_t acc_real_t = 0.0;
	register half_t acc_half_t = 0.0;
	double theshold = -2222;

	const register real_t f = real_t(FMA_INPUT);
	const register half_t i = half_t(FMA_INPUT);

	for (int count = 0; count < NUM_COMPOSE_DIVISOR; count++) {
		acc_real_t = fma_dmr(f, f, acc_real_t);
		acc_half_t = fma_dmr(i, i, acc_half_t);

#if CHECKBLOCK == 1
//		if((count % CHECKBLOCK) == 0) {
//			check_relative_error(acc_half_t, acc_real_t);

			theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

			acc_half_t = half_t(acc_real_t);
//		}
#endif

	}

#if CHECKBLOCK == 0
	check_relative_error(acc_half_t, acc_real_t);
//	theshold = fmax(theshold, fabs(double(acc_real_t) - double(acc_half_t)));

#endif

	if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
		printf("THRESHOLD CHECKBLOCK, %.20e, %d\n", theshold, CHECKBLOCK);
	}

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_half_t;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_real_t;
}

#endif /* DMR_KERNELS_CU_ */
