/*
 * dmr_kernels.cu
 *
 *  Created on: 26/03/2019
 *      Author: fernando
 */

#ifndef DMR_KERNELS_CU_
#define DMR_KERNELS_CU_

__device__ long long errors = 0;

__device__ half inline fma_dmr(half a, half b, half acc) {
	return __hfma(a, b, acc);
}

__device__ float inline fma_dmr(float a, float b, float acc) {
	return __fmaf_rn(a, b, acc);
}

__device__ double inline fma_dmr(double a, double b, double acc) {
	return fma(a, b, acc);
}

__device__ void inline check_relative_error(half acc_incomplete, float acc_full,
		float e) {
	register float relative_error = fabsf(
			acc_full - __half2float(acc_incomplete)) / acc_full;
	if (relative_error > e) {
		atomicAdd(&errors, 1);
	}
}

__device__ void inline check_relative_error(float acc_incomplete,
		double acc_full, double e) {
	register float relative_error = fabs(acc_full - acc_incomplete) / acc_full;
	if (relative_error > e) {
		atomicAdd(&errors, 1);
	}
}



/**
 * In case of DMR is HALF and FLOAT
 * FMA case
 */
template<typename incomplete, typename full>
__global__ void MicroBenchmarkKernel_FMA_DMR(incomplete *d_R0_one,
		full *d_R0_second, full error_threshold) {
	register full acc_full = OUTPUT_R;
	register full input_a_full = INPUT_A;
	register full input_b_full = INPUT_B;
	register full input_a_neg_full = -INPUT_A;
	register full input_b_neg_full = -INPUT_B;

#if defined(test_precision_double)
	register incomplete acc_incomplete = OUTPUT_R;
	register incomplete input_a_incomplete = INPUT_A;
	register incomplete input_b_incomplete = INPUT_B;
	register incomplete input_a_neg_incomplete = -INPUT_A;
	register incomplete input_b_neg_incomplete = -INPUT_B;
#else
	register incomplete acc_incomplete = __float2half_rn(OUTPUT_R);
	register incomplete input_a_incomplete = __float2half_rn(INPUT_A);
	register incomplete input_b_incomplete = __float2half_rn(INPUT_B);
	register incomplete input_a_neg_incomplete = __float2half_rn(-INPUT_A);
	register incomplete input_b_neg_incomplete = __float2half_rn(-INPUT_B);
#endif


#pragma unroll 512
	for (register unsigned int count = 0;
			count < (OPS / (4 * OPS_PER_THREAD_OPERATION)); count++) {
#if defined(test_precision_double)

		acc_full = fma(input_a_full, input_b_full, acc_full);
		acc_full = fma(input_a_neg_full, input_b_full, acc_full);
		acc_full = fma(input_a_full, input_b_neg_full, acc_full);
		acc_full = fma(input_a_neg_full, input_b_neg_full, acc_full);

		acc_incomplete = __fmaf_rn(input_a_incomplete, input_b_incomplete,     acc_incomplete);
		acc_incomplete = __fmaf_rn(input_a_neg_incomplete, input_b_incomplete, acc_incomplete);
		acc_incomplete = __fmaf_rn(input_a_incomplete, input_b_neg_incomplete, acc_incomplete);
		acc_incomplete = __fmaf_rn(input_a_neg_incomplete, input_b_neg_incomplete, acc_incomplete);
#else
		acc_full = __fmaf_rn(input_a_full, input_b_full, acc_full);
		acc_full = __fmaf_rn(input_a_neg_full, input_b_full, acc_full);
		acc_full = __fmaf_rn(input_a_full, input_b_neg_full, acc_full);
		acc_full = __fmaf_rn(input_a_neg_full, input_b_neg_full, acc_full);

		acc_incomplete = __hfma(input_a_incomplete, input_b_incomplete,     acc_incomplete);
		acc_incomplete = __hfma(input_a_neg_incomplete, input_b_incomplete, acc_incomplete);
		acc_incomplete = __hfma(input_a_incomplete, input_b_neg_incomplete, acc_incomplete);
		acc_incomplete = __hfma(input_a_neg_incomplete, input_b_neg_incomplete, acc_incomplete);
#endif

	}

	check_relative_error(acc_incomplete, acc_full, error_threshold);

	d_R0_one[blockIdx.x * blockDim.x + threadIdx.x] = acc_incomplete;
	d_R0_second[blockIdx.x * blockDim.x + threadIdx.x] = acc_full;

}

#endif /* DMR_KERNELS_CU_ */
