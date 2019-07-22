/*
 * GEMMWMMA.h
 *
 *  Created on: 12/08/2018
 *      Author: fernando
 */

#ifndef GEMM_H_
#define GEMM_H_

#include "GEMMBase.h"

template<class real_t>
class GEMM: public GEMMBase<real_t, real_t> {
public:

	GEMM(
			const std::vector<real_t>& host_a0, //Matrix A
			const std::vector<real_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<real_t, real_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {

	}
	void gemm() {
//		sw_mxm_kernel<<<this->grid_dim, this->block_dim, this->shared_memory>>>(
//				this->device_ptr_d0.data(), this->device_ptr_c0.data(),
//				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
//				this->alpha, this->beta, this->k, this->k);
//
//		this->debug("sw_mxm device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
	}
};

template<class half_t, class real_t>
class GEMMDMRMIXED: public GEMMBase<half_t, real_t> {
public:

	GEMMDMRMIXED(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<half_t, real_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {

	}

	void gemm() {
//		this->device_is_memory_bad.clear();
//		sw_mxm_dmr_kernel<<<this->grid_dim, this->block_dim, this->shared_memory>>>(
//				this->device_ptr_d0.data(), this->device_ptr_mixed_dmr.data(),
//				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
//				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
//				this->k);
//
//		this->debug("device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
<<<<<<< HEAD
=======
//
//		int dev = 0;
//		cudaDeviceProp deviceProp;
//		rad::checkFrameworkErrors(cudaGetDeviceProperties(&deviceProp, dev));

//		enum {
//			// Compute the right amount of shared memory to request.
//			// We need shared memory to hold per-CTA C and D matrix tiles, and to cache
//			// per-CTA chunks
//			// of the A and B matrices. Therefore, the right amount to request is the
//			// maximum of those
//			// two numbers.
//			SHMEM_SZ = MAX(
//					sizeof(half_t) * (BLOCK_COL_TILES * M)
//							* (CHUNK_K * K + SKEW_HALF) * 2,
//					M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N
//							* (BLOCK_COL_WARPS * WARP_COL_TILES)
//							* sizeof(real_t))
//		};

		// printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);
//
//		rad::checkFrameworkErrors(
//				cudaFuncSetAttribute(hw_mxm<real_t>,
//						cudaFuncAttributeMaxDynamicSharedMemorySize, this->SHMEM_SZ));
/*
		hw_mxm_kernel<real_t> <<<this->deviceProp.multiProcessorCount,
		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
				this->k);
*/
		this->debug("hw_mxm device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());

	}

	void hw_mxm_dmr() {
		// OPTIMIZED TENSOR + GEMM SW
		hw_mxm_dmr_kernel<half_t, real_t> <<<
				this->deviceProp.multiProcessorCount,
				THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
				this->device_ptr_d1.data(), this->device_ptr_c0.data(),
				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
				this->alpha, this->beta, this->k, this->k);

		this->debug("hw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());

	}

	void sw_mxm() {
		this->device_is_memory_bad.clear();
		dim3 threads(32, 32);

		// sw_mxm_kernel<real_t> <<<this->grid_dim, this->block_dim, this->shared_memory>>>(
		sw_mxm_kernel<real_t> <<<this->grid_dim, this->block_dim, threads>>>(
				this->device_ptr_d0.data(), this->device_ptr_c0.data(),
				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
				this->alpha, this->beta, this->k, this->k);

		this->debug("sw_mxm device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
>>>>>>> eb1f16f8781b5c2314b7ffc5f89951dc4f099d3a
		//end
	}
};

template<class real_t>
class GEMMDMR: public GEMMBase<real_t, real_t> {
public:

<<<<<<< HEAD
	GEMMDMR(
			const std::vector<real_t>& host_a0, //Matrix A
			const std::vector<real_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<real_t, real_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {
=======
		this->debug("sw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		//end
	}
>>>>>>> eb1f16f8781b5c2314b7ffc5f89951dc4f099d3a

	}
	void gemm() {
//		this->device_is_memory_bad.clear();
//		sw_mxm_dmr_kernel<<<this->grid_dim, this->block_dim, this->shared_memory>>>(
//				this->device_ptr_d0.data(), this->device_ptr_mixed_dmr.data(),
//				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
//				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
//				this->k);
//
//		this->debug("device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		//end
	}
};

#endif /* GEMMWMMA_H_ */
