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

void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y = 1,
		size_t z = 1);

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

__device__ float sigmod_gpu(float in);

__device__ float df_sigmod_gpu(float f_x);
__device__ float dot_gpu(float *x, int x_size, float *w);



#endif /* CUDAUTIL_H_ */
