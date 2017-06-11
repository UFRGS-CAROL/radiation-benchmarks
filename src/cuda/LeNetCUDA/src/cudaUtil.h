/*
 * cudaUtil.h
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#ifndef CUDAUTIL_H_
#define CUDAUTIL_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 1024
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

void cuda_gridsize(dim3*, dim3*, size_t, size_t, size_t z);

#endif /* CUDAUTIL_H_ */
