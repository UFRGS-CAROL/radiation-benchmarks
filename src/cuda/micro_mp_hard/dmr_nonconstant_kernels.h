/*
 * dmr_nonconstant_kernels.h
 *
 *  Created on: Jul 30, 2019
 *      Author: fernando
 */

#ifndef DMR_NONCONSTANT_KERNELS_H_
#define DMR_NONCONSTANT_KERNELS_H_

#include <assert.h>

#include "Parameters.h"
#include "device_functions.h"

#ifndef DEFAULT_64_BIT_MASK
#define DEFAULT_64_BIT_MASK 0xffff000000000000
#endif

__device__ double xor_(const double& a, const double& b) {
	long long a_i = __double_as_longlong(a);
	long long b_i = __double_as_longlong(b);
	long long c_i = a_i ^ b_i;
	return __longlong_as_double(c_i);
}

__device__ double uint64_to_double(const uint64& d) {
	return __longlong_as_double(d);
}

__device__ int count_most_signficant_bit(const uint64 bin) {
	assert(sizeof(double) == sizeof(uint64));
	uint64 int_val;
	memcpy(&int_val, &bin, sizeof(double));
	uint64 bit = 0;
	for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
		bit++;
		if (int_val & i) {
			break;
		}
	}

	return bit;
}

__device__ uint64 double_to_uint64(const double& d) {
	return __double_as_longlong(d);
}

__device__ void check_bit_error(const float& lhs, const double rhs,
		uint64 mask = DEFAULT_64_BIT_MASK) {
	double lhs_double = double(lhs);
	uint64 lhs_ll = double_to_uint64(lhs_double);
	uint64 rhs_ll = double_to_uint64(rhs);
	uint64 xor_result = lhs_ll ^ rhs_ll;
	uint64 and_result = xor_result & mask;

	if (and_result != 0) {
		atomicAdd(&errors, 1);
	}

}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_ADDNONCONSTANT(real_t* input,
		real_t* output_real_t, real_t* threshold_out, half_t* output_half_t,
		int num_op) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	register real_t acc_real_t = 0.0;
	register half_t acc_half_t = 0.0;

	register real_t this_thread_input_real_t = input[thread_id];
	register half_t this_thread_input_half_t = half_t(input[thread_id]);
	register real_t threshold;
	for (int count = 0; count < OPS; count++) {
		acc_real_t = add_dmr(this_thread_input_real_t, acc_real_t);
		acc_half_t = add_dmr(this_thread_input_half_t, acc_half_t);

		threshold = acc_real_t - real_t(acc_half_t);
		if ((count % num_op) == 0) {
			check_bit_error(acc_half_t, acc_real_t);
		}
	}
	output_real_t[thread_id] = acc_real_t;
	output_half_t[thread_id] = acc_half_t;
	threshold_out[thread_id] = fabs(threshold);
}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_MULNONCONSTANT(real_t* input,
		real_t* output_real_t, real_t* threshold_out, half_t* output_half_t,
		int num_op) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	register real_t this_thread_input_real_t = input[thread_id];
	register half_t this_thread_input_half_t = half_t(input[thread_id]);

	register real_t acc_real_t = this_thread_input_real_t;
	register half_t acc_half_t = this_thread_input_half_t;

	register real_t threshold;
	for (int count = 0; count < num_op; count++) {
		acc_real_t = mul_dmr(this_thread_input_real_t, acc_real_t);
		acc_half_t = mul_dmr(this_thread_input_half_t, acc_half_t);

		threshold = (acc_real_t) - (real_t(acc_half_t));

	}
	check_bit_error(acc_half_t, acc_real_t);

	output_real_t[thread_id] = acc_real_t;
	output_half_t[thread_id] = acc_half_t;
	threshold_out[thread_id] = fabs(threshold);
}

template<typename half_t, typename real_t>
__global__ void MicroBenchmarkKernel_FMANONCONSTANT(real_t* input,
		real_t* output_real_t, real_t* threshold_out, half_t* output_half_t,
		int num_op) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	register real_t this_thread_input_real_t = input[thread_id];
	register half_t this_thread_input_half_t = half_t(input[thread_id]);

	register real_t acc_real_t = this_thread_input_real_t;
	register half_t acc_half_t = this_thread_input_half_t;

	register real_t threshold;
	for (int count = 0; count < num_op; count++) {
		acc_real_t = fma_dmr(this_thread_input_real_t, this_thread_input_real_t,
				acc_real_t);
		acc_half_t = fma_dmr(this_thread_input_half_t, this_thread_input_half_t,
				acc_half_t);

		threshold = acc_real_t - real_t(acc_half_t);

	}

	check_bit_error(acc_half_t, acc_real_t);

	output_real_t[thread_id] = acc_real_t;
	output_half_t[thread_id] = acc_half_t;
	threshold_out[thread_id] = fabs(threshold);
}

#endif /* DMR_NONCONSTANT_KERNELS_H_ */
