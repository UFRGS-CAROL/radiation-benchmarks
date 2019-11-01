/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

__device__ unsigned long long errors = 0;

__device__ __forceinline__
double abs__(double a) {
	return fabs(a);
}

__device__ __forceinline__
float abs__(float a) {
	return fabsf(a);
}

__device__ __forceinline__
half abs__(half a) {
	return fabsf(a);
}

template<typename real_t> __device__ __forceinline__
void check_relative_error(real_t& lhs, real_t& rhs) {
	real_t diff = abs__(lhs - rhs);

	if (diff > real_t(ZERO_DOUBLE)) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

__device__ __forceinline__
void check_relative_error(float& lhs, double& rhs) {
	const float diff = abs__(__fdividef(lhs, float(rhs)));
	if (diff < MIN_PERCENTAGE && diff > HUNDRED_PERCENT) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

__device__ __forceinline__
void check_relative_error(float& lhs, double& rhs, const uint32_t threshold) {
	const float rhs_as_float = float(rhs);
	const float diff = fabs(lhs - rhs_as_float);
	const float thre = (*(float*) (&threshold));

	if (diff > thre) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */
__device__ __forceinline__
float fma_inline(float a, float b, float acc) {
	return __fmaf_rn(a, b, acc);
}

__device__ __forceinline__
double fma_inline(double a, double b, double acc) {
	return __fma_rn(a, b, acc);
}

__device__ __forceinline__
half fma_inline(half a, half b, half acc) {
	return __hfma(a, b, acc);
}

__device__ __forceinline__
half2 fma_inline(half2 a, half2 b, half2 acc) {
	return __hfma2(a, b, acc);
}

#endif /* DEVICE_FUNCTIONS_H_ */
