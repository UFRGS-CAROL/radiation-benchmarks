/*
 * device_functions.h
 *
 *  Created on: Mar 28, 2019
 *      Author: carol
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "cuda_utils.h"

__device__ __forceinline__ double abs__(double a) {
	return fabs(a);
}

__device__ __forceinline__ float abs__(float a) {
	return fabsf(a);
}

__device__ __forceinline__ half abs__(half a) {
	return fabsf(a);
}

template<typename incomplete, typename full>
__device__ __forceinline__ void check_relative_error(incomplete acc_incomplete,
		full acc_full, full e) {
	register full relative_error = abs__(acc_full - full(acc_incomplete))
			/ acc_full;
	if (relative_error > e) {
		atomicAdd(&errors, 1);
	}
}

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */

__device__ __forceinline__ double fma_dmr(double a, double b, double acc) {
	return fma(a, b, acc);
}

__device__ __forceinline__ float fma_dmr(float a, float b, float acc) {
	return __fmaf_rn(a, b, acc);
}

__device__              __forceinline__ half fma_dmr(half a, half b, half acc) {
	return __hfma(a, b, acc);
}

/**
 * ----------------------------------------
 * ADD DMR
 * ----------------------------------------
 */

__device__ __forceinline__ double add_dmr(double a, double b) {
	return __dadd_rn(a, b);
}

__device__ __forceinline__ float add_dmr(float a, float b) {
	return __fadd_rn(a, b);
}

__device__              __forceinline__ half add_dmr(half a, half b) {
	return __hadd(a, b);
}

/**
 * ----------------------------------------
 * MUL DMR
 * ----------------------------------------
 */

__device__ __forceinline__ double mul_dmr(double a, double b) {
	return __dmul_rn(a, b);
}

__device__ __forceinline__ float mul_dmr(float a, float b) {
	return __fmul_rn(a, b);
}

__device__ __forceinline__ half mul_dmr(half a, half b) {
	return __hmul(a, b);
}

#endif /* DEVICE_FUNCTIONS_H_ */
