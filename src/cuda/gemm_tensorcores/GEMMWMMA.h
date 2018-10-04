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

__device__ unsigned long long int is_memory_bad = 0;

template<class real_t>
__device__ real_t inline read_voter(real_t *v1, real_t *v2, real_t *v3,
		int offset) {

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
		atomicAdd(&is_memory_bad, 1);
	}

	return in1;
}

//__device__ half inline read_voter(half *v1, half *v2, half *v3, int offset) {
//
//	register half in1 = v1[offset];
//	register half in2 = v2[offset];
//	register half in3 = v3[offset];
//
//	if (__heq(in1, in2) || __heq(in1, in3)) {
//		return in1;
//	}
//
//	if (__heq(in2, in3)) {
//		return in2;
//	}
//
//	if (__hne(in1, in2) && __hne(in2, in3) && __hne(in1, in3)) {
//		atomicAdd(&is_memory_bad, 1);
//	}
//
//	return in1;
//}


template<class half_t, class real_t>
__global__ void wmma_matrix_mul(
		half_t *a0, half_t *a1, half_t *a2,
		half_t *b0, half_t *b1, half_t *b2,
		real_t *c0, real_t *c1, real_t *c2,
		real_t*d0, real_t *d1, real_t *d2,
		size_t M, size_t N, size_t K, float alpha, float beta){

	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	register int k;
//	printf("entrou wmma mult");
	register real_t acc = 0.0;
	for (k = 0; k < N; k++) {
		half_t tmp = read_voter(a0, a1, a2, ty * N + k)
						* read_voter(b0, b1, b2, k * N + tx);
		acc = real_t(tmp) + acc;
	}
//	printf("passou for wmma mult");
	acc += read_voter(c0, c1, c2, ty * N + tx);

	d0[ty * N + tx] = acc;
	d1[ty * N + tx] = acc;
	d2[ty * N + tx] = acc;
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

//	//const host_half_t* host_ptr_a1, const host_half_t*host_ptr_a2,
//	//const host_half_t* host_ptr_b1, const host_half_t* host_ptr_b2,
//	//const host_half_t* host_ptr_c1, const host_half_t* host_ptr_c2,
//	GEMMWMMA(const host_half_t* host_ptr_a0, const host_half_t* host_ptr_b0,
//			const real_t* host_ptr_c0, size_t rows_a, size_t cols_a,
//			size_t cols_b);
//
//	virtual ~GEMMWMMA();
//	/**
//	 * Template multiplication
//	 */
//	void mul();
//
//	//const host_half_t* host_ptr_a1, const host_half_t* host_ptr_a2,
//	//const host_half_t* host_ptr_b1, const host_half_t* host_ptr_b2,
//	//const host_half_t* host_ptr_c1, const host_half_t* host_ptr_c2
//
//	void push_arrays(const host_half_t* host_ptr_a0, const host_half_t* host_ptr_b0,
//			const real_t* host_ptr_c0);
//
//	void pull_array(real_t* host_ptr_d0, real_t* host_ptr_d1,
//			real_t* host_ptr_d2);
//
//	void debug(std::string str);
//
//	unsigned long long int get_memory_errors();

	void mul() {
		//		//No double multiplication is allowed
		if (std::is_same<half_t, double>::value
				|| std::is_same<half_t, float>::value) {
			throw "Double/Float multiplication is not allowed with tensor cores, use GEMM base class instead";
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

		printf("entrou funcao mul");
		wmma_matrix_mul<half_t, real_t> <<<gridDim, blockDim>>>(
				this->device_ptr_a0, this->device_ptr_a1, this->device_ptr_a2,
				this->device_ptr_b0, this->device_ptr_b1, this->device_ptr_b2,
				this->device_ptr_c0, this->device_ptr_c1, this->device_ptr_c2,
				this->device_ptr_d0, this->device_ptr_d1, this->device_ptr_d2,
				this->rows_a, this->cols_b, this->rows_b, 1.0, 1.0);

		printf("passou chamada wmma mult");		
		this->debug("device synchronize");
		check_framework_errors(cudaDeviceSynchronize());

		printf("passou sync mul");
		this->byte_size_c = this->rows_c * this->cols_c * sizeof(float);

	}

	GEMMWMMA(const host_half_t* host_ptr_a0, const host_half_t* host_ptr_b0,
			const real_t* host_ptr_c0, size_t rows_a, size_t cols_a,
			size_t cols_b) {

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
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a0, host_ptr_a0,
						this->rows_a * this->cols_a * sizeof(half_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_a0, host_ptr_a0,
						this->rows_a * this->cols_a * sizeof(half_t),
						cudaMemcpyHostToDevice));

		this->debug("memcpy arrays B");

		//PUSH B's
		check_framework_errors(
				cudaMemcpy(this->device_ptr_b0, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_b0, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));

		check_framework_errors(
				cudaMemcpy(this->device_ptr_b0, host_ptr_b0,
						this->rows_b * this->cols_b * sizeof(half_t),
						cudaMemcpyHostToDevice));

		this->debug("memcpy arrays C");
		//PUSH C's
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c0, host_ptr_c0,
						this->rows_c * this->cols_c * sizeof(real_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c0, host_ptr_c0,
						this->rows_c * this->cols_c * sizeof(real_t),
						cudaMemcpyHostToDevice));
		check_framework_errors(
				cudaMemcpy(this->device_ptr_c0, host_ptr_c0,
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
	}

	void debug(std::string str) {
		if (this->to_debug) {
			std::cout << str << std::endl;
		}
	}

	unsigned long long int get_memory_errors() {
		unsigned long long int host_is_memory_bad;
		check_framework_errors(
				cudaMemcpyFromSymbol(&host_is_memory_bad, "is_memory_bad",
						sizeof(unsigned long long int), 0,
						cudaMemcpyDeviceToHost));
		return host_is_memory_bad;
	}

};

#endif /* GEMMWMMA_H_ */
