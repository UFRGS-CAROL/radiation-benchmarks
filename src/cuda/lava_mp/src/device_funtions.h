/*
 * device_funtions.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNTIONS_H_
#define DEVICE_FUNTIONS_H_

#if __CUDA_ARCH__ >= 600
#include <cuda_fp16.h>

#endif
#include "common.h"
#include "block_threshold.h"

__device__ unsigned long long errors;
/**
 * EXP
 */
#if __CUDA_ARCH__ >= 600
__DEVICE_INLINE__
half exp__(half lhs) {
	return hexp(lhs);
}
#endif

__DEVICE_INLINE__
float exp__(float lhs) {
	return expf(lhs);
}

__DEVICE_INLINE__
double exp__(double lhs) {
	return exp(lhs);
}

__DEVICE_INLINE__
void check_max_float(float lhs) {
	float this_block_lower_threshold = atomicExch(
			lower_relative_limit + blockIdx.x,
			lower_relative_limit[blockIdx.x]);
	float this_block_upper_threshold = atomicExch(
			upper_relative_limit + blockIdx.x,
			upper_relative_limit[blockIdx.x]);

	if (lhs > this_block_upper_threshold) {
		atomicExch(lower_relative_limit + blockIdx.x, lhs);
	}

	if (lhs < this_block_lower_threshold) {
		atomicExch(upper_relative_limit + blockIdx.x, lhs);
	}
}

__DEVICE_INLINE__
bool relative_error(float& lhs, double& rhs) {
	float rhs_as_float = float(rhs);
	float relative = __fdividef(lhs, rhs_as_float);

	return (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE);
}

__DEVICE_INLINE__
bool uint_error(float& lhs, double& rhs, uint32_t& threshold,
		uint32_t& sub_res) {
	float rhs_float = float(rhs);
	uint32_t rhs_data = *((uint32_t*) (&rhs_float));
	uint32_t lhs_data = *((uint32_t*) (&lhs));
	sub_res = SUB_ABS(lhs_data, rhs_data);
	return sub_res > threshold;
}

__DEVICE_INLINE__
void check_bit_error(float& lhs, double& rhs, uint32_t threshold) {

#ifdef BUILDRELATIVEERROR
	if (relative_error(lhs, rhs)) {
#else
	uint32_t sub_res;
	if (uint_error(lhs, rhs, thresholds[blockIdx.x], sub_res)) {
		atomicMax(thresholds + blockIdx.x, sub_res);
#endif
		printf("%f\n", __fdividef(lhs, float(rhs)));
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_bit_error(FOUR_VECTOR<float>& lhs, FOUR_VECTOR<double>& rhs,
		const uint32_t threshold) {
	//CHECK each one of the coordinates
	check_bit_error(lhs.v, rhs.v, threshold);
	check_bit_error(lhs.x, rhs.x, threshold);
	check_bit_error(lhs.y, rhs.y, threshold);
	check_bit_error(lhs.z, rhs.z, threshold);
}

template<typename real_t> __DEVICE_INLINE__
void check_bit_error(real_t& lhs, real_t& rhs, const uint32_t threshold = 0) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

inline uint64_t get_dmr_error() {
	uint64_t dmr_errors;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&dmr_errors, errors, sizeof(uint64_t), 0,
					cudaMemcpyDeviceToHost));

	const uint64_t zero_value = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &zero_value, sizeof(uint64_t), 0,
					cudaMemcpyHostToDevice));
	return dmr_errors;
}

#endif /* DEVICE_FUNTIONS_H_ */
