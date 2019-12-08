/*
 * device_functions.h
 *
 *  Created on: Mar 28, 2019
 *      Author: carol
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include "common.h"

__device__ unsigned long long errors = 0;

template<const uint32 THRESHOLD_UINT32, const uint32 COUNT>
__DEVICE__ void check_bit_error(const float& lhs, const double& rhs) {

#if BUILDRELATIVEERROR == 0
	const uint32 lhs_data = __float_as_uint(lhs);
	const uint32 rhs_data = __float_as_uint(float(rhs));
	uint32 sub_res;
	if (lhs_data > rhs_data) {
		sub_res = lhs_data - rhs_data;
	} else {
		sub_res = rhs_data - lhs_data;
	}

	if (sub_res > THRESHOLD_UINT32) {
		atomicAdd(&errors, 1);
	}
#else
	float rhs_as_float = float(rhs);
	float relative = __fdividef(lhs, rhs_as_float);

	switch (COUNT) {
		case 1: {
			constexpr float min_percentage = 1.0f - 1.0e-6;
			constexpr float max_percentage = 1.0f + 1.0e-6;
			if(relative < min_percentage || relative > max_percentage) {
				atomicAdd(&errors, 1);
			}
			return;
		}
		case 10: {
			constexpr float min_percentage = 1.0f - 1.0e-5;
			constexpr float max_percentage = 1.0f + 1.0e-5;
			if(relative < min_percentage || relative > max_percentage) {
				atomicAdd(&errors, 1);
			}
			return;
		}
		case 100: {
			constexpr float min_percentage = 1.0f - 1.0e-4;
			constexpr float max_percentage = 1.0f + 1.0e-4;
			if(relative < min_percentage || relative > max_percentage) {
				atomicAdd(&errors, 1);
			}
			return;
		}
		case 1000: {
			constexpr float min_percentage = 1.0f - 1.0e-3;
			constexpr float max_percentage = 1.0f + 1.0e-3;
			if(relative < min_percentage || relative > max_percentage) {
				atomicAdd(&errors, 1);
			}
			return;
		}
		case OPS: {
			constexpr float min_percentage = 1.0f - 1.0e-2;
			constexpr float max_percentage = 1.0f + 1.0e-2;
			if(relative < min_percentage || relative > max_percentage) {
				atomicAdd(&errors, 1);
			}
			return;
		}

	}

#endif
}

template<const uint32 THRESHOLD_UINT32, typename real_t>
__DEVICE__ void check_bit_error(real_t& lhs, real_t& rhs) {
	if (lhs != rhs) {
		atomicAdd(&errors, 1);
	}
}

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

#if __CUDA_ARCH__ >= 600

__DEVICE__ half fma_dmr(half a, half b, half acc) {
	return __hfma(a, b, acc);
}
#endif
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

#if __CUDA_ARCH__ >= 600

__DEVICE__ half add_dmr(half a, half b) {
	return __hadd(a, b);
}
#endif
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

#if __CUDA_ARCH__ >= 600
__DEVICE__ half mul_dmr(half a, half b) {
	return __hmul(a, b);
}
#endif

#endif /* DEVICE_FUNCTIONS_H_ */
