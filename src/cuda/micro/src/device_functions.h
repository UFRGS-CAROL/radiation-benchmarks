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
template<bool USEFASTMATH, typename real_t>
__DEVICE_INLINE__ real_t fma_inline(real_t a, real_t b, real_t c) {
	return a * b + c;
}

template<>
__DEVICE_INLINE__ double fma_inline<true>(double a, double b, double c) {
	return __fma_rn(a, b, c);

}

template<>
__DEVICE_INLINE__ float fma_inline<true>(float a, float b, float c) {
	return __fmaf_rn(a, b, c);
}

/**
 * ----------------------------------------
 * ADD
 * ----------------------------------------
 */
template<bool USEFASTMATH, typename real_t>
__DEVICE_INLINE__ real_t add_inline(real_t a, real_t b) {
	return a + b;
}

template<>
__DEVICE_INLINE__ double add_inline<true>(double a, double b) {
	return __dadd_rn(a, b);
}

template<>
__DEVICE_INLINE__ float add_inline<true>(float a, float b) {
	return __fadd_rn(a, b);
}

/**
 * ----------------------------------------
 * MUL
 * ----------------------------------------
 */
template<bool USEFASTMATH, typename real_t>
__DEVICE_INLINE__ real_t mul_inline(real_t a, real_t b) {
	return a * b;
}

template<>
__DEVICE_INLINE__ double mul_inline<true>(double a, double b) {
	return __dmul_rn(a, b);
}

template<>
__DEVICE_INLINE__ float mul_inline<true>(float a, float b) {
	return __fmul_rn(a, b);
}

/**
 * ----------------------------------------
 * PYTHAGOREAN IDENTITY
 * ----------------------------------------
 */

template<bool USEFASTMATH, typename real_t>
__DEVICE_INLINE__ real_t pythagorean_identity(real_t a, real_t b) {
	return pow(sin(a), real_t(2.0)) + pow(cos(b), real_t(2.0));
}

template<>
__DEVICE_INLINE__ float pythagorean_identity<false>(float a, float b) {
	return powf(sinf(a), 2.0f) + powf(cosf(b), 2.0f);
}

template<>
__DEVICE_INLINE__ float pythagorean_identity<true>(float a, float b) {
	return __powf(__sinf(a), 2.0f) + __powf(__cosf(b), 2.0f);
}

/**
 * ----------------------------------------
 * EULER
 * ----------------------------------------
 */
//double call
template<bool USEFASTMATH, typename real_t>
__DEVICE_INLINE__ real_t euler(real_t a) {
	return exp(a);
}

template<>
__DEVICE_INLINE__ float euler<false>(float a) {
	return expf(a);
}

template<>
__DEVICE_INLINE__ float euler<true>(float a) {
	return __expf(a);
}

//HALF instructions

#endif /* DEVICE_FUNCTIONS_H_ */
