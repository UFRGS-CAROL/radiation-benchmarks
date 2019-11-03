/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

#define __DEVICE_INLINE__  __device__ __forceinline__

__device__ unsigned long long errors = 0;

__DEVICE_INLINE__
double abs__(double a) {
	return fabs(a);
}

__DEVICE_INLINE__
float abs__(float a) {
	return fabsf(a);
}

__DEVICE_INLINE__
half abs__(half a) {
	return fabsf(a);
}

template<typename real_t>  __DEVICE_INLINE__
void check_relative_error(real_t lhs, real_t rhs, const uint32_t threshold) {
	real_t diff = abs__(lhs - rhs);

	if (diff > real_t(ZERO_DOUBLE)) {
		atomicAdd(&errors, 1);
	}
}

//__DEVICE_INLINE__
//void check_relative_error(float lhs, double rhs) {
//	const float diff = abs__(__fdividef(lhs, float(rhs)));
//	if (diff < MIN_PERCENTAGE && diff > HUNDRED_PERCENT) {
//		atomicAdd(&errors, 1);
//	}
//	lhs = rhs;
//}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs, const uint32_t threshold) {
	float rhs_as_float = float(rhs);
	uint32_t l = __float_as_uint(lhs);
	uint32_t r = __float_as_uint(rhs_as_float);
	uint32_t diff = SUB_ABS(l, r);
	if (diff > threshold) {
		atomicAdd(&errors, 1);
	}
}

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */
__DEVICE_INLINE__
float fma_inline(float a, float b, float acc) {
	return __fmaf_rn(a, b, acc);
}

__DEVICE_INLINE__
double fma_inline(double a, double b, double acc) {
	return __fma_rn(a, b, acc);
}

__DEVICE_INLINE__
half fma_inline(half a, half b, half acc) {
#if __CUDA_ARCH__ >= 600
	return __hfma(a, b, acc);
#else
	return __fmaf_rn(float(a), float(b), float(acc));
#endif
}

//__DEVICE_INLINE__
//half2 fma_inline(half2 a, half2 b, half2 acc) {
//	return __hfma2(a, b, acc);
//}

#endif /* DEVICE_FUNCTIONS_H_ */
