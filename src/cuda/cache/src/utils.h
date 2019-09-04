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


typedef unsigned char byte;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef signed int int32;
typedef signed long long int int64;

#define cuda_check(error) rad::__checkFrameworkErrors(error, __LINE__, __FILE__)

static void error(std::string err) {
	throw std::runtime_error("ERROR:" + err);
}

template<typename T>
inline void copy_to_gpu(char* symbol, T mem) {
	cuda_check(cudaMemcpyToSymbol(symbol, &mem, sizeof(T), 0));
}

template<typename T>
inline T copy_from_gpu(char *symbol) {
	T mem;
	cuda_check(cudaMemcpyFromSymbol(&mem, symbol, sizeof(T), 0));
	return mem;
}

#ifdef __NVCC__
__device__ static void sleep_cuda(int64 clock_count) {
	int64 start = clock64();
	int64 clock_offset = 0;
	while (clock_offset < clock_count) {
		clock_offset = clock64() - start;
	}
}
#endif

#endif /* UTILS_H_ */
