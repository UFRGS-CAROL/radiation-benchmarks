/*
 * GEMMBase.h
 *
 *  Created on: Jul 21, 2019
 *      Author: fernando
 */

#ifndef GEMMBASE_H_
#define GEMMBASE_H_

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

void exception(std::string msg, std::string file, int line) {
	throw std::runtime_error(msg + " at " + file + ":" + std::to_string(line));
}

#define throw_line(msg) exception(msg, __FILE__, __LINE__)

typedef enum {
	NONDMRGEMM, DMRGEMM, DMRGEMMMIXED, NONDMRWMMA, DMRWMA
} GEMMTYPE;

//host_half, half, host_real_t, real_t
template<class half_t, class real_t, class dmr_mixed_real_t>
class GEMMBase {
public:
	GEMMBase(
			const std::vector<half_t>& host_a0, //Matrix A
			const std::vector<half_t>& host_b0, // MAtrix B
			const std::vector<real_t>&host_c0, // Matric C
			const std::vector<real_t>& host_d0, size_t k, real_t alpha,
			real_t beta, GEMMTYPE gemm_type) :
			k(k), alpha(alpha), beta(beta), gemm_type(gemm_type), device_ptr_d0(
					host_d0) // allocating D vectors

	{ //Alpha and Beta
		if (this->k <= 0) {
			throw_line("columns or rows equal to zero, or less than zero");
		}

		this->host_is_memory_bad.push_back(0);
		this->device_is_memory_bad = this->host_is_memory_bad;

		this->debug("device memory allocation and push memory to device");
		this->push_arrays(host_a0, host_b0, host_c0);

		this->device_ptr_mixed_dmr = std::vector < dmr_mixed_real_t
				> (this->device_ptr_d0.size(), 0);

		// Setup execution parameters
		this->debug("thread dim allocation");

		switch (this->gemm_type) {
		case DMRGEMMMIXED:
		case NONDMRGEMM:
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
				cudaFuncSetAttribute(hw_mxm_kernel<half_t, real_t>,
						cudaFuncAttributeMaxDynamicSharedMemorySize,
						this->shared_memory));
		rad::checkFrameworkErrors(
				cudaFuncSetAttribute(hw_mxm_dmr_kernel<real_t>,
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

//		this->device_ptr_mixed_dmr = std::vector < half_t > (host_d0.size());
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

		this->debug("memcpy arrays A");

		//PUSH A
		this->device_ptr_a0 = host_a0;

		this->debug("memcpy arrays B");

		//PUSH B's
		this->device_ptr_b0 = host_b0;

		this->debug("memcpy arrays C");
		//PUSH C's
		this->device_ptr_c0 = host_c0;
	}

	/**
	 * PULL D array to host
	 */

	void pull_array(std::vector<real_t>& host_d0) {

		this->debug("memcpy array D to host");
		// PULL D's
		host_d0 = this->device_ptr_d0.to_vector();
	}

	/**
	 * Destructor for the GEMM class
	 */

	virtual ~GEMMBase() {
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

	virtual void gemm() = 0;

protected:

	// Memory pointers to device and host data
	rad::DeviceVector<half_t> device_ptr_a0;
	rad::DeviceVector<half_t> device_ptr_b0;
	rad::DeviceVector<real_t> device_ptr_c0;
	rad::DeviceVector<real_t> device_ptr_d0;

	rad::DeviceVector<dmr_mixed_real_t> device_ptr_mixed_dmr;

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

};

#endif /* GEMMBASE_H_ */
