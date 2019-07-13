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
				cudaStreamCreateWithFlags(&this->stream,
						cudaStreamNonBlocking));
	}

	CudaStream(const CudaStream& b){
		this->stream = b.stream;
	}

	CudaStream& operator=(const CudaStream&  b){
		this->stream = b.stream;
		return *this;
	}

	virtual ~CudaStream() {
		rad::checkFrameworkErrors(cudaStreamDestroy(this->stream));
	}

	void sync() {
		rad::checkFrameworkErrors(cudaStreamSynchronize(this->stream));
	}
};

typedef enum {
	STATIC, PERSISTENT, GEMM, COUNT

} KernelType;

static std::ostream& operator<<(std::ostream& os, const KernelType& dt) {
	switch (dt) {
	case STATIC:
		os << std::string("NON-PERSISTENT threads");
		break;
	case PERSISTENT:
		os << std::string("PERSISTENT threads");
		break;
	case GEMM:
		os << std::string("cuBLAS kernel");
		break;
	}
	return os;
}

void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
		const std::vector<CudaStream>& streams, KernelType t, dim3 gridDim,
		dim3 blockDim);

#endif /* MATRIXMUL_KERNEL_H_ */
