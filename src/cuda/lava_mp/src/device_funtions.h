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
float atomicMin(float * addr, float value) {
	float old;
	old = (value >= 0) ?
			__int_as_float(atomicMin((int *) addr, __float_as_int(value))) :
			__uint_as_float(
					atomicMax((unsigned int *) addr, __float_as_uint(value)));

	return old;
}

__DEVICE_INLINE__
float atomicMax(float * addr, float value) {
	float old;
	old = (value >= 0) ?
			__int_as_float(atomicMax((int *) addr, __float_as_int(value))) :
			__uint_as_float(
					atomicMin((unsigned int *) addr, __float_as_uint(value)));

	return old;
}

__DEVICE_INLINE__
void relative_error(float& lhs, double& rhs) {
	float rhs_as_float = float(rhs);
	float relative = __fdividef(lhs, rhs_as_float);
	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (relative < lower_relative_limit[tid]) {
		//lower_relative_limit[tid] = relative;
		atomicAdd(&errors, 1);
	}

	if (relative > upper_relative_limit[tid]) {
		//upper_relative_limit[tid] = relative;
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void uint_error(float& lhs, double& rhs, uint32_t& threshold) {
	float rhs_float = float(rhs);
	uint32_t rhs_data = *((uint32_t*) (&rhs_float));
	uint32_t lhs_data = *((uint32_t*) (&lhs));
	uint32_t sub_res = SUB_ABS(lhs_data, rhs_data);

	uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (sub_res > thresholds[tid]) {
		atomicMax(thresholds + tid, sub_res);

		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_bit_error(float& lhs, double& rhs, uint32_t threshold) {
#ifdef BUILDRELATIVEERROR
	relative_error(lhs, rhs);
#else
	uint_error(lhs, rhs, threshold);
#endif
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
