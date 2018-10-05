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
#include <mma.h>
#include <cuda_fp16.h> // For half precision computation
#include <iostream>

// The only dimensions currently supported by WMMA
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

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

template<class real_t>
__device__ real_t inline read_voter(real_t *v1, real_t *v2, real_t *v3,
		int offset, unsigned long long int* is_memory_bad) {

	register real_t in1 = v1[offset];
	register real_t in2 = v2[offset];
	register real_t in3 = v3[offset];

	if (in1 == in2 || in1 == in3) {
		return in1;
	}

	if (in2 == in3) {
		return in2;
	}

	if (in1 != in2 && in2 != in3 && in1 != in3) {
		atomicAdd(is_memory_bad, 1);
	}

	return in1;
}

template<class half_t, class real_t>
__global__ void wmma_matrix_mul(half_t *a0, half_t *a1, half_t *a2, half_t *b0,
		half_t *b1, half_t *b2, real_t *c0, real_t *c1, real_t *c2, real_t*d0,
		real_t *d1, real_t *d2, size_t M, size_t N, size_t K, float alpha,
		float beta, unsigned long long int* is_memory_bad) {

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;

//
//	printf("\n wmma a0= %f  \n", __half2float(a0[ty * N + k]));
//	printf("\n wmma a1= %f  \n", __half2float(a1[ty * N + k]));
//	printf("\n wmma a2= %f  \n", __half2float(a2[ty * N + k]));
//
//	printf("\n wmma b0= %f  \n", __half2float(b0[ty * N + k]));
//	printf("\n wmma b1= %f  \n", __half2float(b1[ty * N + k]));
//	printf("\n wmma b2= %f  \n", __half2float(b2[ty * N + k]));
//
//	printf("\n wmma c0= %f  \n", (c0[ty * N + k]));
//	printf("\n wmma c1= %f  \n", (c1[ty * N + k]));
//	printf("\n wmma c2= %f  \n", (c2[ty * N + k]));
//
//	printf("read a: %f", __half2float(read_voter<half_t>(a0, a1, a2, ty * N + k, is_memory_bad)));

	register real_t acc = 0.0;
	for (k = 0; k < N; k++) {

		half_t tmp = read_voter<half_t>(a0, a1, a2, ty * N + k, is_memory_bad)
				* read_voter<half_t>(b0, b1, b2, k * N + tx, is_memory_bad);
		acc = real_t(tmp) + acc;

	}

	acc += read_voter<real_t>(c0, c1, c2, ty * N + tx, is_memory_bad);

	d0[ty * N + tx] = acc;
	d1[ty * N + tx] = acc;
	d2[ty * N + tx] = acc;

	// for (int i = 0; i < N; i++)
	// 	printf("d0 %f\n", d0[i]);
	// for (int i = 0; i < N; i++)
	// 	printf("d1 %f\n", d1[i]);
	// for (int i = 0; i < N; i++)
	// 	printf("d2 %f\n", d2[i]);

}

#define check_framework_errors(error) __check_framework_errors(error, __LINE__, __FILE__)
#define error(error) __error(error, __LINE__, __FILE__)

template<class host_half_t, class half_t, class real_t>
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

	size_t byte_size_c;

	bool to_debug = false;

	//to check memory errors
	unsigned long long int* device_is_memory_bad = nullptr;

	void mul() {
		//		//No double multiplication is allowed
		if (std::is_same<half_t, double>::value
				|| std::is_same<half_t, float>::value) {
			throw std::runtime_error(
					"Double/Float multiplication is not allowed with tensor cores, use GEMM base class instead\n");
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

		check_framework_errors(cudaMemset(this->device_is_memory_bad, 0x0, sizeof(unsigned long long int)));

		wmma_matrix_mul<half_t, real_t> <<<gridDim, blockDim>>>(
				this->device_ptr_a0, this->device_ptr_a1, this->device_ptr_a2,
				this->device_ptr_b0, this->device_ptr_b1, this->device_ptr_b2,
				this->device_ptr_c0, this->device_ptr_c1, this->device_ptr_c2,
				this->device_ptr_d0, this->device_ptr_d1, this->device_ptr_d2,
				this->rows_a, this->cols_b, this->rows_b, 1.0, 1.0,
				this->device_is_memory_bad);

		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());

		this->byte_size_c = this->rows_c * this->cols_c * sizeof(float);

	}

	GEMMWMMA(const host_half_t* host_ptr_a0,
			const host_half_t* host_ptr_b0, const real_t* host_ptr_c0,
			size_t rows_a, size_t cols_a, size_t cols_b) {

		this->rows_a = rows_a;
		this->cols_a = cols_a;
		this->rows_b = this->cols_a;
		this->cols_b = cols_b;
		this->cols_c = this->rows_a;
		this->rows_c = this->cols_b;

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
			const host_half_t* host_ptr_b0, const real_t* host_ptr_c0) {

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

	void pull_array(real_t* host_ptr_d0, real_t* host_ptr_d1,
			real_t* host_ptr_d2) {

		this->debug("memcpy array D to host");
		// PULL D's

//		printf("Dd= %f",);
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
