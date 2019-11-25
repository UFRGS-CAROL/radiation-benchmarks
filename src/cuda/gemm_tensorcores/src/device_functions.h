/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

#define __DEVICE_INLINE__  __forceinline__  __device__

__device__ unsigned long long errors = 0;

template<typename real_t> __DEVICE_INLINE__
void check_relative_error(real_t lhs, real_t rhs) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs) {
	float relative = __fdividef(lhs, float(rhs));
	if (relative < MIN_PERCENTAGE || relative > MAX_PERCENTAGE) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE_INLINE__
void check_relative_error(float lhs, double rhs, uint32_t threshold) {
	float rhs_as_float = float(rhs);
	uint32_t lhs_data = *((uint32_t*) &lhs);
	uint32_t rhs_data = *((uint32_t*) &rhs_as_float);

	uint32_t diff = SUB_ABS(lhs_data, rhs_data);

	if (diff > threshold) {
		atomicAdd(&errors, 1);
	}
}

//__DEVICE_INLINE__
//double abs__(double a) {
//	return fabs(a);
//}
//
//__DEVICE_INLINE__
//float abs__(float a) {
//	return fabsf(a);
//}
//
//#if __CUDA_ARCH__ > 600
//__DEVICE_INLINE__
//half abs__(half a) {
//	return fabsf(a);
//}
//#endif

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
//template<typename real_t> __DEVICE_INLINE__
//void fma_inline(volatile real_t& a, volatile real_t& b, volatile real_t& acc) {
//	acc += a * b;
//}
//
//
//#if __CUDA_ARCH__ >= 600
//__DEVICE_INLINE__
//half fma_inline(half a, half b, half acc) {
//	return __hfma(a, b, acc);
//}
//#endif
//
//
//__DEVICE_INLINE__
//half2 fma_inline(half2 a, half2 b, half2 acc) {
//	return __hfma2(a, b, acc);
//}
#endif /* DEVICE_FUNCTIONS_H_ */
