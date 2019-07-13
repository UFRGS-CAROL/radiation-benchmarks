/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */

#include <cooperative_groups.h>
#include <stdexcept>
#include "cublas_v2.h"

#include "matrixMul_kernel.h"
#include "persistent_lib.h"

template<typename real_t>
__device__ void process_mxm_ii(real_t *C, real_t *A, real_t *B, int wA,
		int wB) {
	// Handle to thread block group
	cooperative_groups::thread_block cta =
			cooperative_groups::this_thread_block();
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
		cooperative_groups::sync(cta);

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
		cooperative_groups::sync(cta);

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
__global__ void matrixMulCUDAPersistent(real_t* c, real_t* a, real_t* b, int wA,
		int wB, int nStreams) {
	rad::PersistentKernel pk;
	while (pk.keep_working()) {
		pk.wait_for_work();
		if (pk.is_able_to_process()) {
			for (int streamI = 0; streamI < nStreams; streamI++) {
				int ptr_index = streamI * wA * wB;
				real_t* c_i_ptr = c + ptr_index;
				real_t* a_i_ptr = a + ptr_index;
				real_t* b_i_ptr = b + ptr_index;
				process_mxm_ii(c_i_ptr, a_i_ptr, b_i_ptr, wA, wB);

			}
			pk.iteration_finished();
		}
	}
}

void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB,
		const std::vector<CudaStream>& streams, KernelType t, dim3 gridDim,
		dim3 blockDim) {
	auto streamSize = streams.size();
	switch (t) {
	case PERSISTENT: {
		matrixMulCUDAPersistent<<<gridDim, blockDim, 0, streams[0].stream>>>(C,
				A, B, wA, wB, streamSize);
		rad::checkFrameworkErrors (cudaPeekAtLastError());

break;	}
	case STATIC: {
		for (int streamI = 0; streamI < streamSize; streamI++) {
			int ptr_index = streamI * wA * wB;
			float* c_i_ptr = C + ptr_index;
			float* a_i_ptr = A + ptr_index;
			float* b_i_ptr = B + ptr_index;
			matrixMulCUDANonpersistent<<<gridDim, blockDim, 0,
			streams[streamI].stream>>>(c_i_ptr, a_i_ptr, b_i_ptr, wA,
					wB);
		}

		for (auto stream : streams) {
			stream.sync();
		}
		break;
	}

	case GEMM: {

		static cublasHandle_t handle;
		float alpha = 1;
		float beta = 0;
		cublasStatus_t status = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB, wB, &alpha,
				A, wA, B, wB, &beta, C, wB, streamSize);
		break;
	}

}
}

