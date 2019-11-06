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

#if __CUDA_ARCH__ > 600
__DEVICE_INLINE__
half abs__(half a) {
	return fabsf(a);
}
#endif

template<typename real_t>  __DEVICE_INLINE__
void check_relative_error(real_t lhs, real_t rhs, const uint32_t threshold) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs, const uint32_t threshold) {
	float rhs_as_float = float(rhs);
	uint32_t l = *(uint32_t*)(&lhs);
    uint32_t r = *(uint32_t*)(&rhs_as_float);
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
//__DEVICE_INLINE__
//float fma_inline(float a, float b, float acc) {
//	return __fmaf_rn(a, b, acc);
//}
//
//__DEVICE_INLINE__
//double fma_inline(double a, double b, double acc) {
//	return __fma_rn(a, b, acc);
//}
template<typename real_t> __DEVICE_INLINE__
real_t fma_inline(real_t a, real_t b, real_t acc) {
	return (a * b + acc);
}


#if __CUDA_ARCH__ >= 600
__DEVICE_INLINE__
half fma_inline(half a, half b, half acc) {
	return __hfma(a, b, acc);
}
#endif

//__DEVICE_INLINE__
//half2 fma_inline(half2 a, half2 b, half2 acc) {
//	return __hfma2(a, b, acc);
//}

#endif /* DEVICE_FUNCTIONS_H_ */
