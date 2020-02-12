/*
 * device_functions.h
 *
 *  Created on: Mar 28, 2019
 *      Author: carol
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include <cuda_fp16.h>
#include <cstdint>

#include "utils.h"

#define __DEVICE_INLINE__ __forceinline__ __device__

/**
 * ----------------------------------------
 * FMA
 * ----------------------------------------
 */
template<bool USEFASTMATH>
__DEVICE_INLINE__ double fma_inline(double a, double b, double c) {
#if USEFASTMATH == true
	return __fma_rn(a, b, c);
#else
	return a * b + c;
#endif
}

template<bool USEFASTMATH>
__DEVICE_INLINE__ float fma_inline(float a, float b, float c) {
#if USEFASTMATH == true
	return __fmaf_rn(a, b, c);
#else
	return a * b + c;
#endif
}

/**
 * ----------------------------------------
 * ADD
 * ----------------------------------------
 */
template<bool USEFASTMATH>
__DEVICE_INLINE__ double add_inline(double a, double b) {
#if USEFASTMATH == true
	return __dadd_rn(a, b);
#else
	return a + b;
#endif
}

template<bool USEFASTMATH>
__DEVICE_INLINE__ float add_inline(float a, float b) {
#if USEFASTMATH == true
	return __fadd_rn(a, b);
#else
	return a + b;
#endif
}

/**
 * ----------------------------------------
 * MUL
 * ----------------------------------------
 */
template<bool USEFASTMATH>
__DEVICE_INLINE__ double mul_inline(double a, double b) {
#if USEFASTMATH == true
	return __dmul_rn(a, b);
#else
	return a * b;
#endif
}

template<bool USEFASTMATH>
__DEVICE_INLINE__ float mul_inline(float a, float b) {
#if USEFASTMATH == true
	return __fmul_rn(a, b);
#else
	return a * b;
#endif
}

/**
 * ----------------------------------------
 * PYTHAGOREAN IDENTITY
 * ----------------------------------------
 */
template<bool USEFASTMATH>
__DEVICE_INLINE__ float pythagorean_identity(float a, float b) {
#if USEFASTMATH == true
	return __powf(__sinf(a), 2.0f) + __powf(__cosf(b), 2.0f);
#else
	return powf(sinf(a), 2.0f) + powf(cosf(b), 2.0f);
#endif
}

template<bool USEFASTMATH>
__DEVICE_INLINE__ double pythagorean_identity(double a, double b) {
	return pow(sin(a), double(2.0)) + pow(cos(b), double(2.0));
}

/**
 * ----------------------------------------
 * EULER
 * ----------------------------------------
 */
template<bool USEFASTMATH>
__DEVICE_INLINE__ float euler(float a) {
#if USEFASTMATH == true
	return __expf(a);
#else
	return expf(a);
#endif
}

template<bool USEFASTMATH>
__DEVICE_INLINE__ double euler(double a) {
	return exp(a);
}

//HALF instructions

#if __CUDA_ARCH__ >= 600

__DEVICE_INLINE__ half2 fma_inline(half2 a, half2 b, half2 c) {
	return __hfma2(a, b, c);
}

__DEVICE_INLINE__ half2 add_dmr(half2 a, half2 b) {
	return __hadd2(a, b);
}

__DEVICE_INLINE__ half2 mul_dmr(half2 a, half2 b) {
	return __hmul2(a, b);
}
#endif

#endif /* DEVICE_FUNCTIONS_H_ */
