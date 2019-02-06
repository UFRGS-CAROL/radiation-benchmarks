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
#include "kernels.h"
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>


#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// GPU configuration.

#define WARP_SIZE 32

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8


#define checkKernelErrors(expr) do {                                                        \
    expr;                                                                                   \
                                                                                            \
    cudaError_t __err = cudaGetLastError();                                                 \
    if (__err != cudaSuccess) {                                                             \
        printf("Line %d: '%s' failed: %s\n", __LINE__, # expr, cudaGetErrorString(__err));  \
        abort();                                                                            \
    }                                                                                       \
} while(0)


//ERROR functions definitions
void __check_framework_errors(cudaError_t error, int line, const char* file) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, 250, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
	printf("%s - Line: %d at %s\n", errorDescription, line, file);
	exit (EXIT_FAILURE);
}

void __error(const char* error, int line, const char* file) {
	printf("%s - Line: %d at %s\n", error, line, file);
	exit (EXIT_FAILURE);
}

#define check_framework_errors(error) __check_framework_errors(error, __LINE__, __FILE__)
#define error(error) __error(error, __LINE__, __FILE__)

//host_half, half, host_real_t, real_t
template<class host_half_t, class half_t, class host_real_t, class real_t>
class GEMMWMMA {
public:

	// Memory pointers to device and host data
	half_t* device_ptr_a0 = nullptr;
	half_t* device_ptr_a1 = nullptr;
	half_t* device_ptr_a2 = nullptr;

	half_t* device_ptr_b0 = nullptr;
	half_t* device_ptr_b1 = nullptr;
	half_t* device_ptr_b2 = nullptr;

	real_t* device_ptr_c0 = nullptr;
	real_t* device_ptr_c1 = nullptr;
	real_t* device_ptr_c2 = nullptr;

	real_t* device_ptr_d0 = nullptr;
	real_t* device_ptr_d1 = nullptr;
	real_t* device_ptr_d2 = nullptr;

	// Size of the matrix
	size_t cols_a, rows_a;
	size_t cols_b, rows_b;
	size_t cols_c, rows_c;

	float alpha, beta;
	size_t byte_size_c;

	bool to_debug = false;

	//to check memory errors
	unsigned long long int* device_is_memory_bad = nullptr;

	void mul_mxm_triplicated() {

		this->debug("thread dim allocation");
		//		// Setup execution parameters
		// Setup execution parameters
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(std::ceil(this->cols_a / BLOCK_SIZE),
				std::ceil(this->rows_a / BLOCK_SIZE));

		this->debug("matrix multiplication");

		check_framework_errors(
				cudaMemset(this->device_is_memory_bad, 0x0,
						sizeof(unsigned long long int)));

		matrix_mul<half_t, real_t> <<<grid, threads>>>(this->device_ptr_a0,
				this->device_ptr_a1, this->device_ptr_a2, this->device_ptr_b0,
				this->device_ptr_b1, this->device_ptr_b2, this->device_ptr_c0,
				this->device_ptr_c1, this->device_ptr_c2, this->device_ptr_d0,
				this->device_ptr_d1, this->device_ptr_d2, this->rows_a,
				this->cols_b, this->rows_b, this->alpha, this->beta,
				this->device_is_memory_bad);

		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());

	}

	void mul_mxm() {

		this->debug("thread dim allocation");
		//		// Setup execution parameters
		// Setup execution parameters
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		dim3 grid(std::ceil(this->cols_a / BLOCK_SIZE),
				std::ceil(this->rows_a / BLOCK_SIZE));

		this->debug("matrix multiplication");

		check_framework_errors(
				cudaMemset(this->device_is_memory_bad, 0x0,
						sizeof(unsigned long long int)));

		matrix_mul<half_t, real_t> <<<grid, threads>>>(this->device_ptr_a0,
				this->device_ptr_b0, this->device_ptr_c0, this->device_ptr_d0,
				this->rows_a, this->cols_b, this->rows_b, this->alpha,
				this->beta);

		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());

	}

	void mul_wmma() {
		this->debug("thread dim allocation");
		//		// Setup execution parameters
		// First: using WMMA
		dim3 grid_dim;
		dim3 block_dim;

		// block_dim.x must be a multple of warpSize
		// 128x4 means we have 16 warps and a block computes a 64x64 output tile
		block_dim.x = 128;
		block_dim.y = 4;

		grid_dim.x = (this->rows_a + (WMMA_M * block_dim.x / WARP_SIZE - 1))
				/ (WMMA_M * block_dim.x / WARP_SIZE);
		grid_dim.y = (this->cols_a + WMMA_N * block_dim.y - 1)
				/ (WMMA_N * block_dim.y);

		this->debug("matrix multiplication");

		check_framework_errors(
				cudaMemset(this->device_is_memory_bad, 0x0,
						sizeof(unsigned long long int)));

		simple_wmma_gemm<half_t, real_t> <<<grid_dim, block_dim>>>(
				this->device_ptr_a0, this->device_ptr_b0, this->device_ptr_c0,
				this->device_ptr_d0, this->rows_a, this->cols_b, this->cols_c,
				this->alpha, this->beta);

		
		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());
//
//		int dev = 0;
//		cudaDeviceProp deviceProp;
//		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
//
//		enum {
//		    // Compute the right amount of shared memory to request.
//		    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
//		    // per-CTA chunks
//		    // of the A and B matrices. Therefore, the right amount to request is the
//		    // maximum of those
//		    // two numbers.
//			SHMEM_SZ = MAX(
//					sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
//					M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
//					(BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
//		};
//
//		printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);
//		checkCudaErrors(cudaFuncSetAttribute(compute_gemm<half_t, real_t> , cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
////		checkKernelErrors((compute_gemm<half_t, real_t> <<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,SHMEM_SZ>>>
////				(this->device_ptr_a0, this->device_ptr_b0, this->device_ptr_c0,
////				 this->device_ptr_d0, this->alpha, this->beta)));
//		this->debug("device synchronize");
//		check_framework_errors(cudaDeviceSynchronize());

	}


	void mul_wmma_triplicated() {
		this->debug("thread dim allocation");
				//		// Setup execution parameters
				// First: using WMMA
				dim3 grid_dim;
				dim3 block_dim;

				// block_dim.x must be a multple of warpSize
				// 128x4 means we have 16 warps and a block computes a 64x64 output tile
				block_dim.x = 128;
				block_dim.y = 4;

				grid_dim.x = (this->rows_a + (WMMA_M * block_dim.x / WARP_SIZE - 1))
						/ (WMMA_M * block_dim.x / WARP_SIZE);
				grid_dim.y = (this->cols_a + WMMA_N * block_dim.y - 1)
						/ (WMMA_N * block_dim.y);

				this->debug("matrix multiplication");
				
				
				check_framework_errors(
						cudaMemset(this->device_is_memory_bad, 0x0,
								sizeof(unsigned long long int)));			
				

				simple_wmma_gemm<half_t, real_t> <<<grid_dim, block_dim>>>(
						this->device_ptr_d0, this->device_ptr_d1,this->device_ptr_d2,
						this->rows_a, this->cols_b, this->cols_c, this->alpha, this->beta);
				
		
							
//				int dev = 0;
//				cudaDeviceProp deviceProp;
//				checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
//				enum {
//					// Compute the right amount of shared memory to request.
//					// We need shared memory to hold per-CTA C and D matrix tiles, and to cache
//					// per-CTA chunks
//					// of the A and B matrices. Therefore, the right amount to request is the
//					// maximum of those
//					// two numbers.
//					SHMEM_SZ = MAX(
//							sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
//							M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
//							(BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
//				};
//				//checkCudaErrors(cudaFuncSetAttribute(compute_gemm<half_t, real_t> , cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
//				//checkKernelErrors((compute_gemm<half_t, real_t> <<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,SHMEM_SZ>>>
//				//		(this->device_ptr_d0,this->device_ptr_d1,this->device_ptr_d2, this->alpha, this->beta)));
//			
//				this->debug("device synchronize");
//				check_framework_errors(cudaDeviceSynchronize());
			
	}

	GEMMWMMA(const host_half_t* host_ptr_a0, const host_half_t* host_ptr_b0,
			const host_real_t* host_ptr_c0, size_t rows_a, size_t cols_a,
			size_t cols_b, float alpha, float beta) {

		//		//No double multiplication is allowed
		if (std::is_same<half_t, float>::value) {
			throw std::runtime_error(
					"Double/Float multiplication is not allowed with tensor cores, use GEMM base class instead\n");
		}

		this->rows_a = rows_a;
		this->cols_a = cols_a;
		this->rows_b = this->cols_a;
		this->cols_b = cols_b;
		this->cols_c = this->rows_a;
		this->rows_c = this->cols_b;
		this->alpha = alpha;
		this->beta = beta;

		if (rows_a > 0 && cols_a > 0 && cols_b > 0) {
			this->debug("device memory allocation");
			// A's allocation
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_a0),
							this->rows_a * this->cols_a * sizeof(half_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_a1),
							this->rows_a * this->cols_a * sizeof(half_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_a2),
							this->rows_a * this->cols_a * sizeof(half_t)));

			// B's allocation
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_b0),
							this->rows_b * this->cols_b * sizeof(half_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_b1),
							this->rows_b * this->cols_b * sizeof(half_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_b2),
							this->rows_b * this->cols_b * sizeof(half_t)));

			// C's allocation
			//Alloc memory for real_t DOUBLE OR SINGLE
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_c0),
							this->rows_c * this->cols_c * sizeof(real_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_c1),
							this->rows_c * this->cols_c * sizeof(real_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_c2),
							this->rows_c * this->cols_c * sizeof(real_t)));

			// D's allocation
			//Alloc memory for real_t DOUBLE OR SINGLE
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_d0),
							this->rows_c * this->cols_c * sizeof(real_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_d1),
							this->rows_c * this->cols_c * sizeof(real_t)));
			check_framework_errors(
					cudaMalloc(reinterpret_cast<void **>(&this->device_ptr_d2),
							this->rows_c * this->cols_c * sizeof(real_t)));

			//to pull C array directly
			this->byte_size_c = this->rows_c * this->cols_c * sizeof(real_t);

			check_framework_errors(
					cudaMalloc(
							reinterpret_cast<void **>(&this->device_is_memory_bad),
							sizeof(unsigned long long int)));

			this->debug("push memory to device");

			//set 0 to C matrix
			this->push_arrays(host_ptr_a0, host_ptr_b0, host_ptr_c0);
		} else {
			error("columns or rows equal to zero, or less than zero");
		}

	}

	/**
	 * PUSH arrays to gpu and set 0x0 to C matrix
	 */

	void push_arrays(const host_half_t* host_ptr_a0,
			const host_half_t* host_ptr_b0, const host_real_t* host_ptr_c0) {

		this->debug("memset array D");
		//set 0 to C's matrix
		check_framework_errors(
				cudaMemset(this->device_ptr_d0, 0x00,
						this->rows_c * this->cols_c * sizeof(real_t)));
		check_framework_errors(
				cudaMemset(this->device_ptr_d1, 0x00,
						this->rows_c * this->cols_c * sizeof(real_t)));
		check_framework_errors(
				cudaMemset(this->device_ptr_d2, 0x00,
						this->rows_c * this->cols_c * sizeof(real_t)));

		this->debug("memcpy arrays A");

		//PUSH A
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a0, host_ptr_a0,
						this->rows_a * this->cols_a * sizeof(half_t),
						cudaMemcpyHostToDevice));
//		printf("a0 = %f \n", host_ptr_a0[1]);
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a1, host_ptr_a0,
						this->rows_a * this->cols_a * sizeof(half_t),
						cudaMemcpyHostToDevice));
//		printf("a1 = %f \n", host_ptr_a0[1]);
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a2, host_ptr_a0,
						this->rows_a * this->cols_a * sizeof(half_t),
						cudaMemcpyHostToDevice));

		this->debug("memcpy arrays B");

		//PUSH B's
		check_framework_errors(
				cudaMemcpy(this->device_ptr_b0, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_b1, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));

		check_framework_errors(
				cudaMemcpy(this->device_ptr_b2, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));

		this->debug("memcpy arrays C");
		//PUSH C's
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c0, host_ptr_c0,
						this->rows_c * this->cols_c * sizeof(real_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c1, host_ptr_c0,
						this->rows_c * this->cols_c * sizeof(real_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c2, host_ptr_c0,
						this->rows_c * this->cols_c * sizeof(real_t),
						cudaMemcpyHostToDevice));
	}

	/**
	 * PULL D array to host
	 */

	void pull_array(host_real_t* host_ptr_d0, host_real_t* host_ptr_d1,
			host_real_t* host_ptr_d2) {

		this->debug("memcpy array D to host");
		// PULL D's
		check_framework_errors(
				cudaMemcpy(host_ptr_d0, this->device_ptr_d0, this->byte_size_c,
						cudaMemcpyDeviceToHost));
		check_framework_errors(
				cudaMemcpy(host_ptr_d1, this->device_ptr_d1, this->byte_size_c,
						cudaMemcpyDeviceToHost));
		check_framework_errors(
				cudaMemcpy(host_ptr_d2, this->device_ptr_d2, this->byte_size_c,
						cudaMemcpyDeviceToHost));
	}

	/**
	 * Destructor for the GEMM class
	 */

	virtual ~GEMMWMMA() {

		this->debug("destructor");
		// A's destructors
		if (this->device_ptr_a0 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_a0));
		if (this->device_ptr_a1 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_a1));
		if (this->device_ptr_a2 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_a2));

		// B's destructors
		if (this->device_ptr_b0 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_b0));
		if (this->device_ptr_b1 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_b1));
		if (this->device_ptr_b2 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_b2));

		// C's destructors
		if (this->device_ptr_c0 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_c0));
		if (this->device_ptr_c1 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_c1));
		if (this->device_ptr_c2 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_c2));

		// D's destructors
		if (this->device_ptr_d0 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_d0));
		if (this->device_ptr_d1 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_d1));
		if (this->device_ptr_d2 != nullptr)
			check_framework_errors(cudaFree(this->device_ptr_d2));

		//Is memory bad
		if (this->device_is_memory_bad != nullptr) {
			check_framework_errors(cudaFree(this->device_is_memory_bad));

		}
	}

	void debug(std::string str) {
		if (this->to_debug) {
			std::cout << str << std::endl;
		}
	}

	size_t get_memory_errors() {
		size_t host_is_memory_bad;
		check_framework_errors(
				cudaMemcpy(&host_is_memory_bad, this->device_is_memory_bad,
						sizeof(unsigned long long int),
						cudaMemcpyDeviceToHost));
		return host_is_memory_bad;
	}

};

#endif /* GEMMWMMA_H_ */
