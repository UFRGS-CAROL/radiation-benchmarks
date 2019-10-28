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
void check_relative_error(volatile real_t& lhs, volatile  real_t& rhs, const uint32_t threshold) {
	real_t diff = abs__(lhs - rhs);
	if (diff > real_t(ZERO_DOUBLE)) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

__DEVICE_FUNCTION_INLINE__
void check_relative_error(volatile float& lhs, volatile double& rhs, const uint32_t threshold) {
	const float rhs_as_float = float(rhs);
	const uint32_t lhs_data = __float_as_uint(rhs_as_float);
	const uint32_t rhs_data = __float_as_uint(rhs);

	if (SUB_ABS(lhs_data, rhs_data) > threshold) {
		atomicAdd(&errors, 1);
	}
	lhs = rhs;
}

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */
//
//__DEVICE_FUNCTION_INLINE__
//void fma__(const double& a, const double& b, double& acc) {
//	acc = __fma_rn(a, b, acc);
//}
//
//__DEVICE_FUNCTION_INLINE__
//void fma__(const float& a, const float& b, float& acc) {
//	acc = __fmaf_rn(a, b, acc);
//}

__DEVICE_FUNCTION_INLINE__
void fma__(volatile float& a, volatile float& b, float& acc) {
	acc = __fmaf_rn(a, b, acc);
}

__DEVICE_FUNCTION_INLINE__
void fma__(volatile double& a, volatile double& b, volatile double& acc) {
	acc = __fma_rn(a, b, acc);
}


__DEVICE_FUNCTION_INLINE__
void fma__(const half& a, const half& b, half& acc) {
	acc = __hfma(a, b, acc);
}

__DEVICE_FUNCTION_INLINE__
void fma__(const half2& a, const half2& b, half2& acc) {
	acc = __hfma2(a, b, acc);
}

#endif /* DEVICE_FUNCTIONS_H_ */
