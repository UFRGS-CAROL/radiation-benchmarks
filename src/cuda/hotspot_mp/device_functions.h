/*
 * device_functions.h
 *
 *  Created on: 21/05/2019
 *      Author: fernando
 */

#ifndef DEVICE_FUNCTIONS_H_
#define DEVICE_FUNCTIONS_H_

#include <cuda_fp16.h>

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define BLOCK_SIZE 32

#ifdef __NVCC__

__device__ unsigned long long errors;

template<typename T>
__device__   __forceinline__ T smallest() {
	return 0;
}

template<>
__device__   __forceinline__ half smallest<half>() {
	return 0.0001;
}

template<>
__device__ __forceinline__ float smallest<float>() {
	return 0.0000001;
}

template<>
__device__ __forceinline__ double smallest<double>() {
	return 0.0000001;
}

__device__ __forceinline__ double abs__(double a) {
	return fabs(a);
}

__device__ __forceinline__ float abs__(float a) {
	return fabsf(a);
}

__device__    __forceinline__ half abs__(half a) {
	return fabsf(a);
}

template<typename full, typename incomplete>
__device__ __forceinline__ void compare(const full lhs, const incomplete rhs) {
	const full diff = abs__(lhs - full(rhs));
	const full zero = full(0.000000001);
	if (diff > zero) {
		atomicAdd(&errors, 1);
	}
}

unsigned long long copy_errors() {
	unsigned long long errors_host = 0;
	//Copy errors first
	checkFrameworkErrors(
			cudaMemcpyFromSymbol(&errors_host, errors,
					sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost));


	//Reset the errors variable
	checkFrameworkErrors(
				cudaMemcpyToSymbol(errors, 0,
						sizeof(unsigned long long), 0, cudaMemcpyHostToDevice));
	return errors_host;
}

#endif

#endif /* DEVICE_FUNCTIONS_H_ */
