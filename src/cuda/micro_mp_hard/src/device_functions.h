/*
 * device_functions.h
 *
 *  Created on: Mar 28, 2019
 *      Author: carol
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "Parameters.h"

__device__ unsigned long long errors = 0;

__DEVICE__ double abs__(double a) {
	return fabs(a);
}

__DEVICE__ float abs__(float a) {
	return fabsf(a);
}

__DEVICE__ half abs__(half a) {
	return fabsf(a);
}

__DEVICE__ void compare(const float lhs, const half rhs) {
	const float diff = abs__(lhs - float(rhs));
	const float zero = float(ZERO_HALF);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

__DEVICE__ void compare(const double lhs, const float rhs) {
	const double diff = abs__(lhs - double(rhs));
	const double zero = double(ZERO_FLOAT);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

template<typename T>
__DEVICE__ void compare(const T lhs, const T rhs) {
	const T diff = abs__(lhs - rhs);
	const T zero = T(ZERO_FULL);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

template<const uint32 THRESHOLD_UINT32>
__DEVICE__ void check_bit_error(const float lhs, const double rhs) {
	const float rhs_float = float(rhs);

	const uint32* lhs_ptr = (uint32*) &lhs;
	const uint32* rhs_ptr = (uint32*) &rhs_float;
	const uint32 lhs_data = *lhs_ptr;
	const uint32 rhs_data = *rhs_ptr;
	const uint32 sub_res =
			(lhs_data > rhs_data) ?  lhs_data - rhs_data: rhs_data - lhs_data;
	if (sub_res > THRESHOLD_UINT32) {
		atomicAdd(&errors, 1);
	}
}

template<typename incomplete, typename full>
__DEVICE__ void check_relative_error(incomplete acc_incomplete,
		full acc_full) {
	compare(acc_full, acc_incomplete);
}

//template<typename T>
//__DEVICE__ void cast(volatile T& lhs, const T& rhs) {
//	lhs = rhs;
//}

/*
 * __float2half_rd  round-down mode
 * __float2half_rn round-to-nearest-even mode
 * __float2half_ru  round-up mode
 * __float2half_rz round-towards-zero mode
 */
//__DEVICE__ void cast(volatile half& lhs, const float& rhs) {
//	lhs = __float2half_rn(rhs);
//}
/*
 *__double2float_rd Convert a double to a float in round-down mode.
 *__double2float_rn Convert a double to a float in round-to-nearest-even mode.
 *__double2float_ru Convert a double to a float in round-up mode.
 *__double2float_rz Convert a double to a float in round-towards-zero mode.
 */
//__DEVICE__ void cast(volatile float& lhs, const double& rhs) {
//	lhs = __double2float_rn(rhs);
//}
/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */

__DEVICE__ double fma_dmr(double a, double b, double acc) {
	return __fma_rn(a, b, acc);
}

__DEVICE__ float fma_dmr(float a, float b, float acc) {
	return __fmaf_rn(a, b, acc);
}

__DEVICE__ half fma_dmr(half a, half b, half acc) {
	return __hfma(a, b, acc);
}

/**
 * ----------------------------------------
 * ADD DMR
 * ----------------------------------------
 */

__DEVICE__ double add_dmr(double a, double b) {
	return __dadd_rn(a, b);
}

__DEVICE__ float add_dmr(float a, float b) {
	return __fadd_rn(a, b);
}

__DEVICE__ half add_dmr(half a, half b) {
	return __hadd(a, b);
}

/**
 * ----------------------------------------
 * MUL DMR
 * ----------------------------------------
 */

__DEVICE__ double mul_dmr(double a, double b) {
	return __dmul_rn(a, b);
}

__DEVICE__ float mul_dmr(float a, float b) {
	return __fmul_rn(a, b);
}

__DEVICE__ half mul_dmr(half a, half b) {
	return __hmul(a, b);
}

#endif /* DEVICE_FUNCTIONS_H_ */
