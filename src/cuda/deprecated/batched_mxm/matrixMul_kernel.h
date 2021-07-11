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
__device__ void process_mxm_ii(real_t *C, real_t *A, real_t *B, int wA,
		int wB) {
	// Handle to thread block group
//	cooperative_groups::thread_block cta =
//			cooperative_groups::this_thread_block();
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

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
	process_mxm_ii(c, a, b, wA, wB);
}

template<typename real_t>
__global__ void matrixMulCUDAPersistent(real_t** c, real_t** a, real_t** b,
		int wA, int wB, int nStreams) {
	rad::PersistentKernel pk;

	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			for (int i = 0; i < nStreams; i++) {
				process_mxm_ii(c[i], a[i], b[i], wA, wB);
			}
			pk.iteration_finished();
		}
	}
}

template<typename real_t>
__global__ void matrixMulCUDADynamicParallelism(real_t** c, real_t** a,
		real_t** b, dim3 gridDim, dim3 blockDim, int wA, int wB, int nKernels) {
	for (int i = 0; i < nKernels; i++) {
		matrixMulCUDANonpersistent<<<gridDim, blockDim>>>(c[i], a[i], b[i], wA,
				wB);
	}
}

void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
		const std::vector<std::shared_ptr<CudaStream>>& streams, KernelType t,
		dim3 gridDim, dim3 blockDim, std::shared_ptr<CublasHandle>& handle) {
	auto streamSize = streams.size();

	static std::vector<float*> a_array(streamSize);
	static std::vector<float*> b_array(streamSize);
	static std::vector<float*> c_array(streamSize);

	for (int streamI = 0; streamI < streamSize; streamI++) {
		int ptr_index = streamI * wA * wB;
		c_array[streamI] = C + ptr_index;
		a_array[streamI] = A + ptr_index;
		b_array[streamI] = B + ptr_index;
	}

	static rad::DeviceVector<float*> a_array_dev = a_array;
	static rad::DeviceVector<float*> b_array_dev = b_array;
	static rad::DeviceVector<float*> c_array_dev = c_array;

	switch (t) {
	case PERSISTENT: {
		//Persistent case
		matrixMulCUDAPersistent<<<gridDim, blockDim, 0, streams[0]->stream>>>(
				c_array_dev.data(), a_array_dev.data(), b_array_dev.data(), wA,
				wB, streamSize);
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;
		break;
	}
	case STATIC: {
		for (int streamI = 0; streamI < streamSize; streamI++) {
			int ptr_index = streamI * wA * wB;
			float* c_i_ptr = C + ptr_index;
			float* a_i_ptr = A + ptr_index;
			float* b_i_ptr = B + ptr_index;
			matrixMulCUDANonpersistent<<<gridDim, blockDim, 0,
					streams[streamI]->stream>>>(c_i_ptr, a_i_ptr, b_i_ptr, wA,
					wB);
		}

		for (auto stream : streams) {
			stream->sync();
		}
		break;
	}

	case GEMM: {
		float alpha = 1;
		float beta = 0;
		const float** ptr_a = const_cast<const float**>(a_array_dev.data());
		const float** ptr_b = const_cast<const float**>(b_array_dev.data());
		cublasStatus_t status = cublasSgemmBatched(handle->cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N, wA, wB, wB, &alpha, ptr_a, wA, ptr_b,
				wB, &beta,
				//(float * const *)
				c_array_dev.data(), wB, streamSize);
		rad::checkCublasErrors(status);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		break;
	}

	case DYNAMICPARALLELISM: {
		matrixMulCUDADynamicParallelism<<<1, 1, 0, streams[0]->stream>>>(
				c_array_dev.data(), a_array_dev.data(), b_array_dev.data(),
				gridDim, blockDim, wA, wB, streamSize);

		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		;
		break;
	}

	}
}

#endif /* MATRIXMUL_KERNEL_H_ */
