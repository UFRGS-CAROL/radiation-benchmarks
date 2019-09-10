/*
 * utils.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <cuda_runtime.h> // cudaError_t
#include <string> // error message
#include <iostream>

#include <vector>

#include "cuda_utils.h"

#define DEFAULT_INDEX -1;


//typedef unsigned char byte;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef int int32;
typedef long long int int64;

#define cuda_check(error) rad::__checkFrameworkErrors(error, __LINE__, __FILE__)

static void error(std::string err) {
	throw std::runtime_error("ERROR:" + err);
}

#ifdef __NVCC__
__device__ __forceinline__ static void sleep_cuda(const int64& clock_count) {
	const int64 start = clock64();
	while ((clock64() - start) < clock_count);
}


__device__ __forceinline__ void move_cache_line(uint64* dst, uint64* src) {
	dst[0] = src[0];
	dst[1] = src[1];
	dst[2] = src[2];
	dst[3] = src[3];
	dst[4] = src[4];
	dst[5] = src[5];
	dst[6] = src[6];
	dst[7] = src[7];
	dst[8] = src[8];
	dst[9] = src[9];
	dst[10] = src[10];
	dst[11] = src[11];
	dst[12] = src[12];
	dst[13] = src[13];
	dst[14] = src[14];
	dst[15] = src[15];
}

#endif

#endif /* UTILS_H_ */
