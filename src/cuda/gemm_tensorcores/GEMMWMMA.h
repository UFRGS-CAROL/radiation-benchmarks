/*
 * GEMMWMMA.h
 *
 *  Created on: Jul 21, 2019
 *      Author: fernando
 */

#ifndef GEMMWMMA_H_
#define GEMMWMMA_H_

#include "GEMMBase.h"

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

template<class half_t, class real_t>
class GEMMWMMA: public GEMMBase<half_t, real_t, half_t> {
public:

	GEMMWMMA(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<half_t, real_t, half_t>(host_a0, host_b0, host_c0, host_d0,
					k, alpha, beta, gemm_type) {

	}

	void gemm() {
//		 OPTIMIZED TENSOR + GEMM SW
		hw_mxm_kernel<<<this->deviceProp.multiProcessorCount,
		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
				this->k);

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
			GEMMBase<half_t, real_t, half_t>(host_a0, host_b0, host_c0, host_d0,
					k, alpha, beta, gemm_type) {

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
	size_t shared_memory;
	std::vector<CudaStream> two_streams;

	GEMMWMMADMR(
			const std::vector<real_t>& host_a0, //Matrix A
			const std::vector<real_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			GEMMBase<real_t, real_t, real_t>(host_a0, host_b0, host_c0, host_d0,
					k, alpha, beta, gemm_type) {
		this->shared_memory = std::max(
				sizeof(real_t) * (BLOCK_COL_TILES * M)
						* (CHUNK_K * K + SKEW_HALF) * 2,
				M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N
						* (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float));

//		rad::checkFrameworkErrors(
//				cudaFuncSetAttribute(hw_mxm_kernel<real_t, real_t>,
//						cudaFuncAttributeMaxDynamicSharedMemorySize,
//						this->shared_memory));

		this->two_streams.resize(2);

		std::cout << "M: " << M_GLOBAL << " (" << M << " x " << M_TILES << ")"
				<< std::endl;
		std::cout << "N: " << N_GLOBAL << " (" << N << " x " << N_TILES << ")"
				<< std::endl;
		std::cout << "K: " << K_GLOBAL << " (" << K << " x " << K_TILES << ")"
				<< std::endl;
		if (M_GLOBAL != this->k) {
			throw_line("M_GLOBAL AND K sizes must be the same!");
		}

	}

	void gemm() {
		// OPTIMIZED TENSOR + GEMM SW
		//HARDWARE CALL

		dim3 gridDim;
		dim3 blockDim;

		// blockDim.x must be a multple of warpSize
		// 128x4 means we have 16 warps and a block computes a 64x64 output tile
		blockDim.x = 128;
		blockDim.y = 4;

		gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1))
				/ (WMMA_M * blockDim.x / 32);
		gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1)
				/ (WMMA_N * blockDim.y);

		// If enough shared memory available on the GPU use high performant kernel
//		if (this->deviceProp.sharedMemPerMultiprocessor
//				>= this->shared_memory) {
//			hw_mxm_kernel <<<
//					gridDim, blockDim, 0,
//					this->two_streams[0].stream>>>(this->device_ptr_mixed_dmr.data(),
//					this->device_ptr_c0.data(), this->device_ptr_a0.data(),
//					this->device_ptr_b0.data(), this->alpha, this->beta,
//					this->k, this->k);
//		} else {
//			throw_line("NOT SUPPORTED\n");
//		}

		//SOFTWARE CALL
		sw_mxm_kernel<<<this->grid_dim, this->block_dim, 0,
				this->two_streams[1].stream>>>(this->device_ptr_d0.data(),
				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
				this->k);

		for (auto st : this->two_streams)
			st.sync();
		this->debug("hw_mxm_dmr device synchronize");
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
	}
};

#endif /* GEMMWMMA_H_ */
