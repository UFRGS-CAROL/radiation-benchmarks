/*
 * GEMMWMMA.h
 *
 *  Created on: Jul 21, 2019
 *      Author: fernando
 */

#ifndef GEMMWMMA_H_
#define GEMMWMMA_H_

#include "GEMMBase.h"

template<class half_t, class real_t>
class GEMMWMMA: public GEMMBase<half_t, real_t, half_t> {
public:

	GEMMWMMA(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<half_t, real_t, half_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {


	}

	void gemm() {
		// OPTIMIZED TENSOR + GEMM SW
		//		hw_mxm_dmr_kernel<<<this->deviceProp.multiProcessorCount,
		//		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
		//				this->device_ptr_mixed_dmr.data(), this->device_ptr_c0.data(),
		//				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
		//				this->alpha, this->beta, this->k, this->k);

		this->debug("hw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
	}

};

template<class half_t, class real_t>
class GEMMWMMAMIXED: public GEMMBase<half_t, real_t, half_t> {
public:

	GEMMWMMAMIXED(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<half_t, real_t, half_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {

	}

	void gemm() {
		// OPTIMIZED TENSOR + GEMM SW
		//		hw_mxm_dmr_kernel<<<this->deviceProp.multiProcessorCount,
		//		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
		//				this->device_ptr_mixed_dmr.data(), this->device_ptr_c0.data(),
		//				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
		//				this->alpha, this->beta, this->k, this->k);

		this->debug("hw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
	}

};

template<class real_t>
class GEMMWMMADMR: public GEMMBase<real_t, real_t, real_t> {
public:

	GEMMWMMADMR(
			const std::vector<real_t>& host_a0, //Matrix A
			const std::vector<real_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<real_t, real_t, real_t>(host_a0, host_b0, host_c0, host_d0, k,
					alpha, beta, gemm_type) {

	}
	void gemm() {
		// OPTIMIZED TENSOR + GEMM SW
		//		hw_mxm_dmr_kernel<<<this->deviceProp.multiProcessorCount,
		//		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
		//				this->device_ptr_mixed_dmr.data(), this->device_ptr_c0.data(),
		//				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
		//				this->alpha, this->beta, this->k, this->k);

		this->debug("hw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
	}
};

#endif /* GEMMWMMA_H_ */
