/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include "device_vector.h"
#include "sgemm_nn_64_16_16_16_4.h"
#include <cassert>
#include <vector>
#include <iostream>

typedef float real_t;
// typedef float half_real_t;

void gemm_host(std::vector<real_t>& a, std::vector<real_t>& b,
		std::vector<real_t>& c, real_t alpha, real_t beta, int m, int n,
		int k) {

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			real_t sum = 0;
			for (int p = 0; p < k; p++) {
				sum += a[i * m + p] * b[p * n + j];
			}
			c[i * m + j] = alpha * sum + beta * c[i * m + j];
		}
	}
}

int main(int argc, char **argv) {

	int m;
	int n;
	int k;
	m = n = k = 4096;
	int lda = m;
	int ldb = n;
	int ldc = k;
	real_t alpha = 0.1;
	real_t beta = 0.3;
	const std::vector<real_t> zero_vector(m * k, 0.0);
	const std::vector<half_real_t> zero_vector_inc(m * k, 0.0);
	std::vector<real_t> host_a(m * n, alpha);
	std::vector<real_t> host_b(n * k, beta);
	std::vector<real_t> host_c(m * k, 0.0);
	std::vector<half_real_t> host_c_inc(m * k, 0.0);

	rad::DeviceVector<real_t> device_c(host_c), device_a(host_a), device_b(
			host_b);
	rad::DeviceVector<half_real_t> device_c_inc(host_c_inc);

	cudaStream_t st;
	cudaStreamCreate(&st);
	assert(m > 512 && n > 512 && m % 64 == 0 && n % 16 == 0 && k % 16 == 0);


	for (int t = 0; t < 10; t++) {
		device_c = zero_vector;
		device_c_inc = zero_vector_inc;
		sgemm(st, device_c.data(), device_a.data(), device_b.data(), m, n, k,
				lda, ldb, ldc, alpha, beta);
		// device_c = zero_vector;
		// device_c_inc = zero_vector_inc;
		// sgemm_dmr(st, device_c.data(), device_c_inc.data(), device_a.data(),
		// 		device_b.data(), m, n, k, lda, ldb, ldc, alpha, beta);

	}


	host_c = device_c.to_vector();
	host_c_inc = device_c_inc.to_vector();

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_c[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
	// std::cout << "Incomplete type" << std::endl;
	// for (int i = 0; i < 10; i++) {
	// 	for (int j = 0; j < 10; j++) {
	// 		std::cout << host_c_inc[i * m + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }

	// std::vector<real_t> test_gemm(m * k, 0);
	// gemm_host(host_a, host_b, test_gemm, alpha, beta, m, n, k);

	// for (int i = 0; i < test_gemm.size(); i++) {
	// 	auto g = test_gemm[i], f = host_c[i];
	// 	if (g != f) {
	// 		std::cout << "HOST " << g << " GPU " << f << std::endl;
	// 		break;
	// 	}
	// }
	cudaStreamDestroy(st);
	return 0;
}
