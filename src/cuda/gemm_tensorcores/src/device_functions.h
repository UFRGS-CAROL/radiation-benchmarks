/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

#define __DEVICE_FUNCTION_INLINE__ __device__ __forceinline__

__device__ unsigned long long errors = 0;

__DEVICE_FUNCTION_INLINE__
double abs__(double a) {
	return fabs(a);
}

__DEVICE_FUNCTION_INLINE__
float abs__(float a) {
	return fabsf(a);
}

__DEVICE_FUNCTION_INLINE__
half abs__(half a) {
	return fabsf(a);
}

template<typename real_t> __DEVICE_FUNCTION_INLINE__
void check_relative_error(real_t& lhs, real_t& rhs) {
	real_t diff = abs__(lhs - rhs);

	if (diff > real_t(ZERO_DOUBLE)) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

__DEVICE_FUNCTION_INLINE__
void check_relative_error(float& lhs, double& rhs) {
	const float diff = abs__(__fdividef(lhs, float(rhs)));
	if (diff < MIN_PERCENTAGE && diff > HUNDRED_PERCENT) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

__DEVICE_FUNCTION_INLINE__
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
__DEVICE_FUNCTION_INLINE__
void fma__(float& a, float& b, float& acc) {
	acc = __fmaf_rn(a, b, acc);
}

__DEVICE_FUNCTION_INLINE__
void fma__(double& a, double& b, volatile double& acc) {
	acc = __fma_rn(a, b, acc);
}

__DEVICE_FUNCTION_INLINE__
void fma__(half& a, half& b, half& acc) {
	acc = __hfma(a, b, acc);
}

__DEVICE_FUNCTION_INLINE__
void fma__(half2& a, half2& b, half2& acc) {
	acc = __hfma2(a, b, acc);
}

#endif /* DEVICE_FUNCTIONS_H_ */
