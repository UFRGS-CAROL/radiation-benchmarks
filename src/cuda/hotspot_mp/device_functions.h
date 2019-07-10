/*
 * device_functions.h
 *
 *  Created on: 21/05/2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include <cuda_fp16.h>
#include "cuda_utils.h"

#define ZERO_FLOAT 1e-6
#define ZERO_HALF 1e-4

#ifdef __NVCC__




__device__ unsigned long long errors;

__device__ __forceinline__ double abs__(double a) {
	return fabs(a);
}

__device__ __forceinline__ float abs__(float a) {
	return fabsf(a);
}

__device__    __forceinline__ half abs__(half a) {
	return fabsf(a);
}

template<typename full>
__device__ __forceinline__ void compare(const full lhs, const full rhs) {
	const full diff = abs__(lhs - rhs);
	const full zero = 0.0;
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
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

unsigned long long copy_errors() {
	unsigned long long errors_host = 0;
	//Copy errors first
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&errors_host, errors,
					sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));

	unsigned long long temp = 0;
	//Reset the errors variable
	rad::checkFrameworkErrors(
				cudaMemcpyToSymbol(errors, &temp,
						sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
	return errors_host;
}

#endif

#endif /* DEVICE_FUNCTIONS_H_ */
