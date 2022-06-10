/*
 * GemmCaller.h
 *
 *  Created on: 07/11/2020
 *      Author: fernando
 */

#ifndef GEMMCALLER_H_
#define GEMMCALLER_H_

#include "include/device_vector.h"

#include "no_tensor_kernels.h"

template<typename T>
using DevVec = rad::DeviceVector<T>;

template<const uint32_t COUNT, typename half_t, typename real_t>
struct GemmCaller {
	bool duplicated;
	dim3 dim_grid, dim_block;

	virtual ~GemmCaller() = default;
	virtual void gemm(
			DevVec<real_t>& a_dev, 			//A matrix
			DevVec<real_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<half_t>& c_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) = 0;

	virtual void memcpy_half_t_mem(std::vector<half_t>& host,
			DevVec<half_t>& device) = 0;

	GemmCaller(uint32_t m, uint32_t n) :
			duplicated(false) {
		uint32_t grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		uint32_t grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		this->dim_grid = dim3(grid_cols, grid_rows);
		this->dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
	}
};

template<typename real_t>
struct UnhardenedGemmCaller: public GemmCaller<0, real_t, real_t> {

	void gemm(
			DevVec<real_t>& a_dev, 			//A matrix
			DevVec<real_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<real_t>& c_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(),//a
				b_dev.data(),//b
				c_dev.data(),//c
				alpha, beta, wA, wB);
	}

	void memcpy_half_t_mem(std::vector<real_t>& host, DevVec<real_t>& device) {}

	UnhardenedGemmCaller(uint32_t m, uint32_t n) :
	GemmCaller<0, real_t, real_t>(m, n) {
		std::cout << this->dim_block << std::endl;
		std::cout << this->dim_grid << std::endl;
	} //default constructor

};

template<typename real_t>
struct CUBLASGemmCaller: public GemmCaller<0, real_t, real_t> {
	cublasHandle_t blas_handle;

	~CUBLASGemmCaller() {
		rad::checkCublasErrors(cublasDestroy(this->blas_handle));
	}

	void gemm(
			DevVec<double>& a_dev, 			//A matrix
			DevVec<double>& b_dev, 			//B matrix
			DevVec<double>& c_dev, 			//C matrix
			DevVec<double>& c_dev_half_t,  	//D_Half matrix
			double alpha, double beta, int wA, int wB,
			const uint32_t threshold) {
		rad::checkCublasErrors(
				cublasDgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, a_dev.data(), wA, b_dev.data(),
						wB, &beta, c_dev.data(), wB));
	}

	void gemm(DevVec<float>& a_dev, 			//A matrix
			DevVec<float>& b_dev, 			//B matrix
			DevVec<float>& c_dev, 			//C matrix
			DevVec<float>& c_dev_half_t,  	//D_Half matrix
			float alpha, float beta, int wA, int wB, const uint32_t threshold) {
		rad::checkCublasErrors(
				cublasSgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, a_dev.data(), wA, b_dev.data(),
						wB, &beta, c_dev.data(), wB));
	}

	void gemm(DevVec<half>& a_dev, 			//A matrix
			DevVec<half>& b_dev, 			//B matrix
			DevVec<half>& c_dev, 			//C matrix
			DevVec<half>& c_dev_half_t,  	//D_Half matrix
			half alpha, half beta, int wA, int wB, const uint32_t threshold) {
#if (__CUDACC_VER_MAJOR__ >= 10)

		rad::checkCublasErrors(
				cublasHgemm(this->blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, wA, wB,
						wA, &alpha, a_dev.data(), wA, b_dev.data(),
						wB, &beta, c_dev.data(), wB));
#endif
	}

	void memcpy_half_t_mem(std::vector<real_t>& host, DevVec<real_t>& device) {
	}

	CUBLASGemmCaller(uint32_t m, uint32_t n, bool use_tensor_cores = false) :
			GemmCaller<0, real_t, real_t>(m, n) {
		rad::checkCublasErrors(cublasCreate(&this->blas_handle));

		if (!use_tensor_cores) {
#if (__CUDACC_VER_MAJOR__ >= 10)
			rad::checkCublasErrors(cublasSetMathMode(this->blas_handle, CUBLAS_PEDANTIC_MATH));
#endif
		}else{
            rad::checkCublasErrors(cublasSetMathMode(this->blas_handle, CUBLAS_TENSOR_OP_MATH));
        }
	} //default constructor
};

template<const uint32_t COUNT, typename half_t, typename real_t>
struct DMRMixedGemmCaller: public GemmCaller<COUNT, half_t, real_t> {

	void gemm(
			DevVec<real_t>& a_dev, 			//A matrix
			DevVec<real_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<half_t>& c_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		matrix_mult_kernel_dmr_mixed<COUNT> <<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(),//a
				b_dev.data(),//b
				c_dev.data(),//c
				c_dev_half_t.data(),//d
				alpha, beta, wA, wB, threshold);

	}

	DMRMixedGemmCaller(uint32_t m, uint32_t n) :
	GemmCaller<COUNT, half_t, real_t>(m, n) {
		this->duplicated = true;
	}

	void memcpy_half_t_mem(std::vector<half_t>& host, DevVec<half_t>& device) {
		device.to_vector(host);
	}
};

template<typename real_t>
struct DMRGemmCaller: public GemmCaller<0, real_t, real_t> {

	void gemm(
			DevVec<real_t>& a_dev, 			//A matrix
			DevVec<real_t>& b_dev, 			//B matrix
			DevVec<real_t>& c_dev, 			//C matrix
			DevVec<real_t>& c_dev_half_t,  	//D_Half matrix
			real_t alpha, real_t beta, int wA, int wB,
			const uint32_t threshold) {
		/*matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>( //call
		 a_dev.data(),//a
		 b_dev.data(),//b
		 c_dev.data(),//c
		 alpha, beta, wA, wB);
		 matrix_mult_kernel_unhardened<<<this->dim_grid, this->dim_block>>>(//call
		 a_dev.data(),//a
		 b_dev.data(),//b
		 c_dev_half_t.data(),//c
		 alpha, beta, wA, wB);

		 rad::checkFrameworkErrors(cudaDeviceSynchronize());
		 ;
		 rad::checkFrameworkErrors(cudaPeekAtLastError());
		 ;
		 uint32_t thread_block = BLOCK_SIZE * BLOCK_SIZE;
		 uint32_t grid_block = (wA * wB) / thread_block;
		 compare_two_outputs<<<grid_block, thread_block>>>(c_dev.data(),
		 c_dev_half_t.data());
		 */
		matrix_mult_kernel_dmr<34> <<<this->dim_grid, this->dim_block>>>( //call
				a_dev.data(),//a
				b_dev.data(),//b
				c_dev.data(),//c
				c_dev_half_t.data(),//d
				alpha, beta, wA, wB);

	}

	DMRGemmCaller(uint32_t m, uint32_t n) :
	GemmCaller<0, real_t, real_t>(m, n) {
		this->duplicated = true;
	}

	void memcpy_half_t_mem(std::vector<real_t>& host, DevVec<real_t>& device) {
		device.to_vector(host);
	}
};

#endif /* GEMMCALLER_H_ */
