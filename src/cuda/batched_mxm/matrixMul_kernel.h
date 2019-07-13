/*
 * matrixMul_kernel.h
 *
 *  Created on: 12/07/2019
 *      Author: fernando
 */

#ifndef MATRIXMUL_KERNEL_H_
#define MATRIXMUL_KERNEL_H_


// CUDA runtime
#include <cuda_runtime.h>
#include "cuda_utils.h"

#include <vector>
/**
 * Only to manage cuda Streams
 */
struct CudaStream {
	cudaStream_t stream;
	CudaStream() {
		rad::checkFrameworkErrors(
				cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking));
	}

	virtual ~CudaStream() {
		rad::checkFrameworkErrors(cudaStreamDestroy(this->stream));
	}

	void sync(){
		rad::checkFrameworkErrors(cudaStreamSynchronize(this->stream));
	}
};

typedef enum {
	STATIC,
	PERSISTENT,
	GEMM,
	COUNT
}KernelType;

void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
		const std::vector<CudaStream>& streams, KernelType t, dim3 gridDim,
		dim3 blockDim);

#endif /* MATRIXMUL_KERNEL_H_ */
