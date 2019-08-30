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

#include <iostream>

#include "cublas_v2.h"

#include "persistent_lib.h"

//#include <cooperative_groups.h>
#include <stdexcept>

#include "device_vector.h"

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

	virtual ~CudaStream() {
		rad::checkFrameworkErrors(cudaStreamDestroy(this->stream));
	}

	void sync() {
		rad::checkFrameworkErrors(cudaStreamSynchronize(this->stream));
	}
};

/**
 * Only to manage cuda handle
 */
struct CublasHandle {
	cublasHandle_t cublas_handle;
	CublasHandle() {
		rad::checkCublasErrors(cublasCreate(&this->cublas_handle));
	}

	virtual ~CublasHandle() {
		rad::checkCublasErrors(cublasDestroy(this->cublas_handle));
	}
};

typedef enum {
	STATIC, PERSISTENT, GEMM, DYNAMICPARALLELISM, COUNT

} KernelType;

static std::ostream& operator<<(std::ostream& os, const KernelType& dt) {
	switch (dt) {
	case STATIC:
		os << std::string("NON-PERSISTENT threads");
		return os;
	case PERSISTENT:
		os << std::string("PERSISTENT threads");
		return os;
	case GEMM:
		os << std::string("cuBLAS kernel");
		return os;
	case DYNAMICPARALLELISM:
		os << std::string("Dynamic parallelism");
		return os;

	}
	return os;

}

static const KernelType kernel_types[COUNT] = { STATIC, PERSISTENT, GEMM,
		DYNAMICPARALLELISM };

template<typename real_t>
__device__ void process_mxm_ii(real_t *C, real_t *A, real_t *B, int wA, int wB,
		const dim3 blIdx) {
	// Handle to thread block group
//	cooperative_groups::thread_block cta =
//			cooperative_groups::this_thread_block();
	// Block index
//	int bx = blockIdx.x;
//	int by = blockIdx.y;
	int bx = blIdx.x;
	int by = blIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	real_t Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ real_t As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ real_t Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
//		cooperative_groups::sync(cta);
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();

	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

template<typename real_t>
__global__ void matrixMulCUDANonpersistent(real_t* c, real_t* a, real_t* b,
		int wA, int wB) {
	dim3 block(blockIdx.x, blockIdx.y);
	process_mxm_ii(c, a, b, wA, wB, block);
}

template<typename real_t>
__global__ void matrixMulCUDAPersistent(real_t* c, real_t* a, real_t* b, int wA,
		int wB, int nStreams, const dim3* block_list, int block_slice) {

	rad::PersistentKernel pk;

	//split the work between threads
	int start_block = blockIdx.x;

	while (pk.keep_working()) {
		//printf("AQUI\n");
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			for (int i = 0; i < block_slice; i++) {
				process_mxm_ii(c, a, b, wA, wB, block_list[start_block * block_slice + i]);
			}
			pk.iteration_finished();
		}

	}
}

void matrixMulCUDA(float *C, float *A, float *B, int& wA, int& wB,
		std::vector<std::shared_ptr<CudaStream>>& streams, KernelType& t,
		dim3& gridDim, dim3& blockDim, std::shared_ptr<CublasHandle>& handle,
		dim3* bll = NULL, int block_slice = 0) {
	auto streamSize = streams.size();

	switch (t) {
	case PERSISTENT: {
		//Persistent case
		//std::cout << "before kernel\n";
		matrixMulCUDAPersistent<<<gridDim, blockDim, 0, streams[0]->stream>>>(C,
				A, B, wA, wB, streamSize, bll, block_slice);
		//std::cout << "after kernel\n";
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;
		break;
	}
	case STATIC: {
		matrixMulCUDANonpersistent<<<gridDim, blockDim, 0, streams[0]->stream>>>(
				C, A, B, wA, wB);

		streams[0]->sync();
		break;
	}

	case GEMM: {
		float alpha = 1;
		float beta = 0;
		cublasStatus_t status = cublasSgemm(handle->cublas_handle, CUBLAS_OP_N,
				CUBLAS_OP_N, wA, wB, wB, &alpha, A, wA, B, wB, &beta, C, wB);
		rad::checkCublasErrors(status);
		rad::checkFrameworkErrors (cudaDeviceSynchronize());;
		break;
	}

	case DYNAMICPARALLELISM: {
		std::runtime_error("NOT IMPLMENTED");
		break;
	}

	}
}

struct th_par {
	float *C;
	float *A;
	float *B;
	int wA;
	int wB;
	std::vector<std::shared_ptr<CudaStream>>* streams;
	KernelType t;
	dim3 gridDim, blockDim;
	std::shared_ptr<CublasHandle> handle;
	dim3* bl_list;
	int strea_slice;
};

void thread_call(th_par* th_ptr) {
	matrixMulCUDA(th_ptr->C, th_ptr->A, th_ptr->B, th_ptr->wA, th_ptr->wB,
			*th_ptr->streams, th_ptr->t, th_ptr->gridDim, th_ptr->blockDim,
			th_ptr->handle, th_ptr->bl_list, th_ptr->strea_slice);
	rad::checkFrameworkErrors (cudaDeviceSynchronize());}

#endif /* MATRIXMUL_KERNEL_H_ */
