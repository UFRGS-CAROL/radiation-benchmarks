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
		//end
	}
};

template<class real_t>
class GEMMDMR: public GEMMBase<real_t, real_t> {
public:

	GEMMDMR(
			const std::vector<real_t>& host_a0, //Matrix A
			const std::vector<real_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<real_t, real_t>(host_a0, host_b0, host_c0, host_d0, k,
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
		//end
	}
};

#endif /* GEMMWMMA_H_ */
