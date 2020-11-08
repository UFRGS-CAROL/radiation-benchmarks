/*
 * GemmCallerMMA.h
 *
 *  Created on: 07/11/2020
 *      Author: fernando
 */

#ifndef GEMMCALLERMMA_H_
#define GEMMCALLERMMA_H_

#include <iostream>

#include "GemmCaller.h"
#include "include/cuda_utils.h"
#include "tensor_kernels.h"

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

template<typename half_t, typename real_t = half_t>
struct TensorCoresCaller {
	bool duplicated;
	dim3 dim_grid, dim_block;

//	enum {
	// Compute the right amount of shared memory to request.
	// We need shared memory to hold per-CTA C and D matrix tiles, and to cache per-CTA chunks
	// of the A and B matrices. Therefore, the right amount to request is the maximum of those
	// two numbers.
	int SHMEM_SZ =
			MAX(
					sizeof(half_t) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
					M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(real_t));
//	};

	virtual ~TensorCoresCaller() = default;
	virtual void gemm(
			DevVec<half_t>& a_dev, 			//A matrix
			DevVec<half_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<real_t>& d_dev, 			//D matrix
			DevVec<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) = 0;

	virtual void memcpy_half_t_mem(std::vector<half_t>& host,
			DevVec<real_t>& device) = 0;

	TensorCoresCaller(uint32_t m, uint32_t n) :
			duplicated(false) {

		auto device_prop = rad::get_device();
		// If enough shared memory available on the GPU use high performance kernel
		if (device_prop.sharedMemPerMultiprocessor < SHMEM_SZ) {
			throw_line("Not enough shared memory availiable on the GPU");
		}

		std::cout
				<< "Computing... using high performance kernel compute_gemm \n";

		this->dim_grid = device_prop.multiProcessorCount;
		this->dim_block = THREADS_PER_BLOCK;

#if __CUDA_ARCH__ >= 600
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(matrix_mult_kernel_wmma_unhardened<half_t, real_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
		;
#endif
	}
};

template<typename half_t, typename real_t = half_t>
struct UnhardenedTensorCoresCaller: public TensorCoresCaller<half_t, real_t> {
	void gemm(
			DevVec<half_t>& a_dev, 			//A matrix
			DevVec<half_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<real_t>& d_dev, 			//D matrix
			DevVec<real_t>& d_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {

		matrix_mult_kernel_wmma_unhardened<<<this->dim_grid, this->dim_block, this->SHMEM_SZ>>>(
				a_dev.data(), b_dev.data(), c_dev.data(), d_dev.data(), alpha,
				beta);
	}

	void memcpy_half_t_mem(std::vector<real_t>& host, DevVec<real_t>& device) {}

	UnhardenedTensorCoresCaller(uint32_t m, uint32_t n) :
	TensorCoresCaller<half_t, real_t>(m, n) {
		std::cout << this->dim_block << std::endl;
		std::cout << this->dim_grid << std::endl;
	} //default constructor
};

template<typename half_t>
struct DMRTensorCoresCaller: public UnhardenedGemmCaller<half_t>,
		public UnhardenedTensorCoresCaller<half_t> {

	using UnhardenedTensorCoresCaller<half_t, half_t>::duplicated;
	using UnhardenedTensorCoresCaller<half_t, half_t>::gemm;

	void gemm(
			DevVec<half_t>& a_dev, 			//A matrix
			DevVec<half_t>& b_dev, 			//B matrix
			DevVec<half_t>& c_dev, 			//C matrix
			DevVec<half_t>& d_dev, 			//D matrix
			DevVec<half_t>& d_dev_half_t,  	//D_Half matrix
			half_t alpha, half_t beta, int wA, int wB,
			const uint32_t threshold) {

		if (float(beta) != 0.0f) {
			throw_line("This implementation does not support beta != 0.0");
		}

		UnhardenedGemmCaller<half_t>::gemm(a_dev, b_dev, d_dev, d_dev, alpha,
				beta, wA, wB, threshold);

		UnhardenedTensorCoresCaller<half_t>::gemm(a_dev, b_dev, c_dev,
				d_dev_half_t, d_dev_half_t, alpha, beta, wA, wB, threshold);
	}

	void memcpy_half_t_mem(std::vector<half_t>& host, DevVec<half_t>& device) {
		device.to_vector(host);
	}

	DMRTensorCoresCaller(uint32_t m, uint32_t n) :
			UnhardenedGemmCaller<half_t>(m, n), UnhardenedTensorCoresCaller<
					half_t>(m, n) {
		this->duplicated = true;

	} //default constructor
};

#endif /* GEMMCALLERMMA_H_ */
