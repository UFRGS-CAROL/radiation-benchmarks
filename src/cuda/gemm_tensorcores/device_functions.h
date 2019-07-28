/*
 * device_functions.h
 *
 *  Created on: Jul 27, 2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#define ZERO_FULL 1e-13

#ifndef ZERO_FLOAT
#define ZERO_FLOAT 2.2e-05
#endif

#ifndef ZERO_HALF
#define ZERO_HALF 4.166E-5
#endif

#ifndef ZERO_DOUBLE
#define ZERO_DOUBLE 0.0
#endif

__device__ unsigned long long errors = 0;

__device__ __forceinline__ double abs__(double a) {
	return fabs(a);
}

__device__ __forceinline__ float abs__(float a) {
	return fabsf(a);
}

__device__      __forceinline__ half abs__(half a) {
	return fabsf(a);
}

__device__ __forceinline__ void compare(const float lhs, const half rhs) {
	const float diff = abs__(lhs - float(rhs));
	const float zero = float(ZERO_HALF);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

__device__ __forceinline__ void compare(const double lhs, const float rhs) {
	const double diff = abs__(lhs - double(rhs));
	const double zero = double(ZERO_FLOAT);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

template<typename T>
__device__ __forceinline__ void compare(const T lhs, const T rhs) {
	const T diff = abs__(lhs - rhs);
	const T zero = T(ZERO_FULL);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

template<typename incomplete, typename full>
__device__ __forceinline__ void check_relative_error(incomplete acc_incomplete,
		full acc_full) {
	compare(acc_full, acc_incomplete);
}

__host__ unsigned long long dmr_errors() {
	unsigned long long ret = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&ret, errors, sizeof(unsigned long long), 0));

	unsigned long long tmp = 0;
	rad::checkFrameworkErrors(
			cudaMemcpyToSymbol(errors, &tmp, sizeof(unsigned long long), 0));

	return ret;
}

/**
 * ----------------------------------------
 * FMA DMR
 * ----------------------------------------
 */

__device__ __forceinline__ void fma_dmr(double& a, double& b, double& acc) {
	acc = __fma_rn(a, b, acc);
}

__device__ __forceinline__ void fma_dmr(float& a, float& b, float& acc) {
	acc = __fmaf_rn(a, b, acc);
}

__device__ __forceinline__ void fma_dmr(half& a, half& b, half& acc) {
	acc = __hfma(a, b, acc);
}

#endif /* DEVICE_FUNCTIONS_H_ */
