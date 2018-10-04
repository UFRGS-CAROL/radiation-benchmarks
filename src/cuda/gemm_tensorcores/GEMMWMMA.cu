/*
 * GEMMWMMA.cu
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#include "GEMMWMMA.h"
#include <mma.h>
// For half precision computation
#include <cuda_fp16.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
// half_t is always half precision
// real_t is always double or single
template<class half_t, class real_t>
__global__ void wmma_matrix_mul(half_t *a, half_t *b, real_t *c, size_t M,
		size_t N, size_t K) {
	// Leading dimensions. Packed with no transpositions.
	int lda = M;
	int ldb = K;
	int ldc = M;

	// Tile using a 2D grid
	int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
	int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

	// Declare the fragments
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> a_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
			half_t, nvcuda::wmma::col_major> b_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			float> acc_frag;

	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
			float> c_frag;

	nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

	// Loop over k
	for (int i = 0; i < K; i += WMMA_K) {
		int aRow = warpM * WMMA_M;
		int aCol = i;

		int bRow = i;
		int bCol = warpN * WMMA_N;

		// Bounds checking
		if (aRow < M && aCol < K && bRow < K && bCol < N) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
			nvcuda::wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

		}
	}

	// Load in the current value of c, scale it by beta, and add this our result scaled by alpha
	int cRow = warpM * WMMA_M;
	int cCol = warpN * WMMA_N;

	if (cRow < M && cCol < N) {
		nvcuda::wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc,
				nvcuda::wmma::mem_col_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc,
				nvcuda::wmma::mem_col_major);
	}
}

namespace radiation {


template<class host_half_t, class half_t, class real_t>
void GEMMWMMA::mul() {
//		//No double multiplication is allowed
	if (std::is_same<T, double>::value) {
		error(
				"Double multiplication is not allowed with tensor cores, use GEMM base class instead");
	}

	this->debug("thread dim allocation");
//		// Setup execution parameters
	dim3 gridDim;
	dim3 blockDim;

	// blockDim.x must be a multple of warpSize
	// 128x4 means we have 16 warps and a block computes a 64x64 output tile
	blockDim.x = 128;
	blockDim.y = 4;

	gridDim.x = (this->rows_a + (WMMA_M * blockDim.x / 32 - 1))
			/ (WMMA_M * blockDim.x / 32);

	gridDim.y = (this->cols_b + WMMA_N * blockDim.y - 1)
			/ (WMMA_N * blockDim.y);

	this->debug("matrix multiplication");
	wmma_matrix_mul<half_t, real_t> <<<gridDim, blockDim>>>(this->device_ptr_a,
			this->device_ptr_b, this->device_ptr_c, this->rows_a, this->cols_b,
			this->rows_b);

	this->debug("device synchronize");
	check_framework_errors (cudaDeviceSynchronize());

this	->byte_size_c = this->rows_c * this->cols_c * sizeof(float);

}

template<class host_half_t, class half_t, class real_t>
GEMMWMMA::GEMMWMMA(const host_half_t* host_ptr_a, const host_half_t* host_ptr_b,
		const host_half_t* host_ptr_c, size_t rows_a, size_t cols_a,
		size_t cols_b) {

	this->rows_a = rows_a;
	this->cols_a = cols_a;
	this->rows_b = this->cols_a;
	this->cols_b = cols_b;
	this->cols_c = this->rows_a;
	this->rows_c = this->cols_b;

	if (rows_a > 0 && cols_a > 0 && cols_b > 0) {
		this->debug("device memory allocation");
		check_framework_errors(
				cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_a),
						this->rows_a * this->cols_a * sizeof(half_t)));
		check_framework_errors(
				cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_b),
						this->rows_b * this->cols_b * sizeof(half_t)));

		//Alloc memory for real_t DOUBLE OR SINGLE
		check_framework_errors(
				cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_c),
						this->rows_c * this->cols_c * sizeof(real_t)));

		check_framework_errors(
				cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_d),
						this->rows_c * this->cols_c * sizeof(real_t)));

		//to pull C array directly
		this->byte_size_c = this->rows_c * this->cols_c * sizeof(real_t);

		this->debug("push memory to device");
		//set 0 to C matrix
		this->push_arrays(host_ptr_a, host_ptr_b);
	} else {
		error("columns or rows equal to zero, or less than zero");
	}

}

/**
 * PUSH arrays to gpu and set 0x0 to C matrix
 */
template<class host_half_t, class half_t, class real_t>
void GEMMWMMA::push_arrays(const host_half_t* host_ptr_a,
		const host_half_t* host_ptr_b, const host_half_t* host_ptr_c) {

	this->debug("memset array D");
	//set 0 to C matrix
	check_framework_errors(
			cudaMemset(this->device_ptr_d, 0x00,
					this->rows_c * this->cols_c * sizeof(half_t)));

	this->debug("memcpy array A");
	//PUSH A
	check_framework_errors(
			cudaMemcpy(this->device_ptr_a, host_ptr_a,
					this->rows_a * this->cols_a * sizeof(half_t),
					cudaMemcpyHostToDevice));

	this->debug("memcpy array B");
	//PUSH B
	check_framework_errors(
			cudaMemcpy(this->device_ptr_b, host_ptr_b,
					this->rows_b * this->cols_b * sizeof(half_t),
					cudaMemcpyHostToDevice));

	this->debug("memcpy array C");
	//PUSH B
	check_framework_errors(
			cudaMemcpy(this->device_ptr_c, host_ptr_c,
					this->rows_c * this->cols_c * sizeof(half_t),
					cudaMemcpyHostToDevice));
}

/**
 * PULL C array to host
 */
template<class host_half_t, class half_t, class real_t>

void GEMMWMMA::pull_array(host_half_t* host_ptr_d) {

	this->debug("memcpy array C to host");
	// PULL C
	check_framework_errors(
			cudaMemcpy(host_ptr_d, this->device_ptr_c, this->byte_size_c,
					cudaMemcpyDeviceToHost));
}

/**
 * Destructor for the GEMM class
 */
template<class host_half_t, class half_t, class real_t>

virtual GEMMWMMA::~GEMMWMMA() {

	this->debug("destructor");
	if (this->device_ptr_a != nullptr)
		check_framework_errors(cudaFree(this->device_ptr_a));

	if (this->device_ptr_b != nullptr)
		check_framework_errors(cudaFree(this->device_ptr_b));

	if (this->device_ptr_c != nullptr)
		check_framework_errors(cudaFree(this->device_ptr_c));

	if (this->device_ptr_d != nullptr)
		check_framework_errors(cudaFree(this->device_ptr_c));
}

}//END NAMESPACE RADIATION
