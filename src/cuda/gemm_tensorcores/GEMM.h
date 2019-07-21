/*
 * GEMMWMMA.h
 *
 *  Created on: 12/08/2018
 *      Author: fernando
 */

#ifndef GEMMWMMA_H_
#define GEMMWMMA_H_

#include <type_traits>
#include <string>
#include <cstdio>
#include <iostream>
#include <vector>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

//CHECK FRAMEWORK ERRORS
#include "cuda_utils.h"

//DEVICE VECTOR
#include "device_vector.h"

//#include "kernels.h"
#include "dmr_kernels.h"

void __error(const char* error, int line, const char* file) {
	printf("%s - Line: %d at %s\n", error, line, file);
	exit (EXIT_FAILURE);
}

#define error(error) __error(error, __LINE__, __FILE__)

typedef enum {
	NONDMR, DMRGEMM, NONDMRWMMA, DMRWMA
} GEMMTYPE;

//host_half, half, host_real_t, real_t
template<class half_t, class real_t>
class GEMM {
public:

	// Memory pointers to device and host data
	rad::DeviceVector<half_t> device_ptr_a0;
	rad::DeviceVector<half_t> device_ptr_a1;
	rad::DeviceVector<half_t> device_ptr_a2;

	rad::DeviceVector<half_t> device_ptr_b0;
	rad::DeviceVector<half_t> device_ptr_b1;
	rad::DeviceVector<half_t> device_ptr_b2;

	rad::DeviceVector<real_t> device_ptr_c0;
	rad::DeviceVector<real_t> device_ptr_c1;
	rad::DeviceVector<real_t> device_ptr_c2;

	rad::DeviceVector<real_t> device_ptr_d0;
	rad::DeviceVector<real_t> device_ptr_d1;

	rad::DeviceVector<real_t> device_ptr_d2;

	// Size of the matrix
	// Only square matrices now
	size_t k;
	real_t alpha, beta;

	bool to_debug = false;

	GEMMTYPE gemm_type;

	//to check memory errors
	rad::DeviceVector<unsigned long long int> device_is_memory_bad;
	std::vector<unsigned long long int> host_is_memory_bad;

	dim3 block_dim;
	dim3 grid_dim;

	cudaDeviceProp deviceProp;

	size_t shared_memory;

	GEMM(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			alpha(alpha), beta(beta), gemm_type(gemm_type), device_ptr_d0(
					host_d0), device_ptr_d1(host_d0), device_ptr_d2(host_d0) // allocating D vectors

	{ //Alpha and Beta
		if (this->k <= 0) {
			error("columns or rows equal to zero, or less than zero");
		}

		this->host_is_memory_bad.push_back(0);
		this->device_is_memory_bad = this->host_is_memory_bad;

		this->debug("device memory allocation and push memory to device");
		this->push_arrays(host_a0, host_b0, host_c0);

		// Setup execution parameters
		this->debug("thread dim allocation");

		switch (this->gemm_type) {
		case NONDMR:
		case DMRGEMM:
			this->block_dim.x = BLOCK_SIZE;
			this->block_dim.y = BLOCK_SIZE;
			this->grid_dim.x = this->k / this->block_dim.x;
			this->grid_dim.y = this->k / this->block_dim.y;
			break;
		case NONDMRWMMA:
		case DMRWMA:
			this->block_dim.x = WMMA_M; //128;
			this->block_dim.y = WMMA_N;

			this->grid_dim.x = (this->k
					+ (WMMA_M * this->block_dim.x / WARP_SIZE - 1))
					/ (WMMA_M * this->block_dim.x / WARP_SIZE);
			this->grid_dim.y = (this->k + WMMA_N * this->block_dim.y - 1)
					/ (WMMA_N * this->block_dim.y);
			break;
		}

		// Compute the right amount of shared memory to request.
		// We need shared memory to hold per-CTA C and D matrix tiles, and to cache
		// per-CTA chunks
		// of the A and B matrices. Therefore, the right amount to request is the
		// maximum of those
		// two numbers.
		this->shared_memory = std::max(
				sizeof(half_t) * (BLOCK_COL_TILES * M)
						* (CHUNK_K * K + SKEW_HALF) * 2,
				M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N
						* (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(real_t));

		//SET ALL THE KERNELS TO USE THE MAXIMUM OF SHARED MEMORY
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(hw_mxm_kernel<half_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						this->shared_memory));
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(hw_mxm_dmr_kernel<half_t, real_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						this->shared_memory));
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(sw_mxm_kernel<half_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						this->shared_memory));
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(sw_mxm_dmr_kernel<half_t, real_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						this->shared_memory));

		int dev = 0;
		rad::checkFrameworkErrors(
				cudaGetDeviceProperties(&this->deviceProp, dev));
	}

	/**
	 * PUSH arrays to gpu and set 0x0 to C matrix
	 */

	void push_arrays(const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // Matrix B
			const std::vector<real_t>&host_c0) { //Matrix C

		this->debug("memset array D");
		//set 0 to C's matrix
		this->device_ptr_d0.clear();
		this->device_ptr_d1.clear();
		this->device_ptr_d2.clear();

		this->debug("memcpy arrays A");

		//PUSH A
		this->device_ptr_a0 = host_a0;
		this->device_ptr_a1 = host_a0;
		this->device_ptr_a2 = host_a0;

		this->debug("memcpy arrays B");

		//PUSH B's
		this->device_ptr_b0 = host_b0;
		this->device_ptr_b1 = host_b0;
		this->device_ptr_b2 = host_b0;

		this->debug("memcpy arrays C");
		//PUSH C's
		this->device_ptr_c0 = host_c0;
		this->device_ptr_c1 = host_c0;
		this->device_ptr_c2 = host_c0;
	}

	/**
	 * PULL D array to host
	 */

	void pull_array(std::vector<real_t>& host_d0, std::vector<real_t>& host_d1,
			std::vector<real_t>& host_d2) {

		this->debug("memcpy array D to host");
		// PULL D's
		host_d0 = this->device_ptr_d0.to_vector();
		host_d1 = this->device_ptr_d1.to_vector();
		host_d2 = this->device_ptr_d2.to_vector();
	}

	/**
	 * Destructor for the GEMM class
	 */

	virtual ~GEMM() {
		this->debug("destructor");
	}

	void debug(std::string str) {
		if (this->to_debug) {
			std::cout << str << std::endl;
		}
	}

	size_t get_memory_errors() {
		this->host_is_memory_bad = this->device_is_memory_bad.to_vector();
		return this->host_is_memory_bad[0];
	}

	void gemm() {
		switch (this->gemm_type) {
		case NONDMR:
			this->sw_mxm();
			break;
		case DMRGEMM:
			this->sw_mxm_dmr();
			break;
		case NONDMRWMMA:
			this->hw_mxm();
			break;
		case DMRWMA:
			this->hw_mxm_dmr();
			break;
		}
	}

private:

	void hw_mxm() {
		this->debug("matrix multiplication");

		this->device_is_memory_bad.clear();

//		this->debug("device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
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
		hw_mxm_kernel<real_t> <<<this->deviceProp.multiProcessorCount,
		THREADS_PER_BLOCK, this->shared_memory>>>(this->device_ptr_d0.data(),
				this->device_ptr_c0.data(), this->device_ptr_a0.data(),
				this->device_ptr_b0.data(), this->alpha, this->beta, this->k,
				this->k);

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

		sw_mxm_kernel<real_t> <<<this->grid_dim, this->block_dim, this->shared_memory>>>(
				this->device_ptr_d0.data(), this->device_ptr_c0.data(),
				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
				this->alpha, this->beta, this->k, this->k);

		this->debug("sw_mxm device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		//end
	}

	void sw_mxm_dmr() {
		this->device_is_memory_bad.clear();
		sw_mxm_dmr_kernel<half_t, real_t> <<<this->grid_dim, this->block_dim,
				this->shared_memory>>>(this->device_ptr_d0.data(),
				this->device_ptr_d1.data(), this->device_ptr_c0.data(),
				this->device_ptr_a0.data(), this->device_ptr_b0.data(),
				this->alpha, this->beta, this->k, this->k);

		this->debug("device synchronize");
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		//end
	}

	/**
	 * DEPRECATED
	 */
//	void mul_mxm_triplicated() {
//
//		this->debug("thread dim allocation");
//		//		// Setup execution parameters
//		// Setup execution parameters
//		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//		dim3 grid(std::ceil(this->cols_a / BLOCK_SIZE),
//				std::ceil(this->rows_a / BLOCK_SIZE));
//
//		this->debug("matrix multiplication");
//
//		rad::checkFrameworkErrors(
//				cudaMemset(this->device_is_memory_bad, 0x0,
//						sizeof(unsigned long long int)));
//
//		// matrix_mul<half_t, real_t> <<<grid, threads>>>(this->device_ptr_a0,
//		// 		this->device_ptr_a1, this->device_ptr_a2, this->device_ptr_b0,
//		// 		this->device_ptr_b1, this->device_ptr_b2, this->device_ptr_c0,
//		// 		this->device_ptr_c1, this->device_ptr_c2, this->device_ptr_d0,
//		// 		this->device_ptr_d1, this->device_ptr_d2, this->rows_a,
//		// 		this->cols_b, this->rows_b, this->alpha, this->beta,
//		// 		this->device_is_memory_bad);
//
//		this->debug("device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
//
//	}
//	void mul_mxm() {
//
//
//		this->debug("thread dim allocation");
//		//		// Setup execution parameters
//		// Setup execution parameters
//		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
//		dim3 grid(std::ceil(this->cols_a / BLOCK_SIZE),
//				std::ceil(this->rows_a / BLOCK_SIZE));
//
//		this->debug("matrix multiplication");
//
//		rad::checkFrameworkErrors(
//				cudaMemset(this->device_is_memory_bad, 0x0,
//						sizeof(unsigned long long int)));
//
//		// matrix_mul<half_t, real_t> <<<grid, threads>>>(this->device_ptr_a0,
//		// 		this->device_ptr_b0, this->device_ptr_c0, this->device_ptr_d0,
//		// 		this->rows_a, this->cols_b, this->rows_b, this->alpha,
//		// 		this->beta);
//
//
//		// matrix_mul_dmr<half_t, real_t> <<<grid, threads>>>(this->device_ptr_a0, this->device_ptr_a1,
//		// 		this->device_ptr_b0, this->device_ptr_c0, this->device_ptr_d0,this->device_ptr_d1,
//		// 		this->rows_a, this->cols_b, this->rows_b, this->alpha,
//		// 		this->beta);
//
//
//
//		this->debug("device synchronize");
//		rad::checkFrameworkErrors(cudaDeviceSynchronize());
//
//	}
//	void mul_gemm() {
//		 this->debug("thread dim allocation");
//		 //		// Setup execution parameters
//		 		// First: using WMMA
//		 		dim3 grid_dim;
//		 		dim3 block_dim;
//
//		 		// block_dim.x must be a multple of warpSize
//		 		// 128x4 means we have 16 warps and a block computes a 64x64 output tile
//		 		block_dim.x = WMMA_M; //128;
//		     	block_dim.y = WMMA_N; //4;
//
//		 		grid_dim.x = (this->rows_a + (WMMA_M * block_dim.x / WARP_SIZE - 1))
//		 				/ (WMMA_M * block_dim.x / WARP_SIZE);
//		 		grid_dim.y = (this->cols_a + WMMA_N * block_dim.y - 1)
//		 				/ (WMMA_N * block_dim.y);
//
//		 		this->debug("matrix multiplication");
//
//		 		rad::checkFrameworkErrors(
//		 				cudaMemset(this->device_is_memory_bad, 0x0,
//		 						sizeof(unsigned long long int)));
//
//		 		// simple_gemm<half_t, real_t> <<<grid_dim, block_dim>>>(this->device_ptr_a0, this->device_ptr_b0, this->device_ptr_c0,
//		  	// 	this->device_ptr_d0, this->rows_a, this->cols_b, this->cols_c,
//		  	// 	this->alpha, this->beta);
//
//	}
//	void mul_gemm_wmma_triplicated() {
//		this->debug("thread dim allocation");
//		//		// Setup execution parameters
//		// First: using WMMA
//		dim3 grid_dim;
//		dim3 block_dim;
//
//		// block_dim.x must be a multple of warpSize
//		// 128x4 means we have 16 warps and a block computes a 64x64 output tile
//		block_dim.x = 128;
//		block_dim.y = 4;
//
//		grid_dim.x = (this->rows_a + (WMMA_M * block_dim.x / WARP_SIZE - 1))
//				/ (WMMA_M * block_dim.x / WARP_SIZE);
//		grid_dim.y = (this->cols_a + WMMA_N * block_dim.y - 1)
//				/ (WMMA_N * block_dim.y);
//
//		this->debug("matrix multiplication");
//
//		rad::checkFrameworkErrors(
//				cudaMemset(this->device_is_memory_bad, 0x0,
//						sizeof(unsigned long long int)));
//
//		// s_tensor_gemm_triplicate<half_t, real_t> <<<grid_dim, block_dim>>>(
//		// 		this->device_ptr_d0, this->device_ptr_d1,this->device_ptr_d2,
//		// 		this->rows_a, this->cols_b, this->cols_c, this->alpha, this->beta);
//
//		//OPTIMIZED TENSOR GEMM NOT IMPLEMENTED
//	}
};

#endif /* GEMMWMMA_H_ */