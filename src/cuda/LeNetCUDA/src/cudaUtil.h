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

#define BLOCK_SIZE 32


inline void __cudaSafeCall(cudaError err, const char *file, const int line);
inline void __cudaCheckError(const char *file, const int line);

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )


void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y = 1,
		size_t z = 1);


inline void __cudaSafeCall(cudaError err, const char *file, const int line) {

	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	return;
}

inline void __cudaCheckError(const char *file, const int line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
				cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}

	return;
}


#endif /* CUDAUTIL_H_ */

