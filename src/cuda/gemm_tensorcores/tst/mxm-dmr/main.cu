#include "kernels.h"
#include <cassert>
#include <vector>
#include <iostream>
#include <random>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CHECK_BLOCK 32
#define THRESHOLD 1

typedef half real_t;
typedef float half_t;




int main(int argc, char **argv) {

	int m;
	int n;
	int k;
	m = n = k = 4096;

	/*
	std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<real_t> distr(-10000.00, 10000.00); 
	*/
  	const real_t alpha = 1.1f;
  	const real_t beta = 1.2f;
  	std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<half_t> distr(0, 100); 

	real_t* host_a = (real_t*)calloc(m * k, sizeof(real_t));
	real_t* host_b = (real_t*)calloc(k * n, sizeof(real_t));
	real_t* host_c = (real_t*)calloc(m * n, sizeof(real_t));
	real_t* host_d = (real_t*)calloc(m * n, sizeof(real_t));
	half_t* host_d_half = (half_t*)calloc(m * n, sizeof(half_t));

	for (int i = 0; i < m * k; i++) host_a[i] = (real_t*)distr(eng);
	for (int i = 0; i < m * k; i++) host_b[i] = (real_t*)distr(eng);
	for (int i = 0; i < m * k; i++) host_c[i] = 0;	
	
	real_t *device_a, *device_b, *device_c, *device_d;
	half_t  *device_d_half;
	cudaMalloc((void**)&device_a, m * k * sizeof(real_t));
	cudaMalloc((void**)&device_b, k * n * sizeof(real_t));
	cudaMalloc((void**)&device_c, m * n * sizeof(real_t));
	cudaMalloc((void**)&device_d, m * n * sizeof(real_t));
	cudaMalloc((void**)&device_d_half, m * n * sizeof(half_t));


	cudaMemcpy(device_a, host_a, m * k * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, k * n * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_c, host_c, m * n * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_d, host_d, m * n * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_d_half, host_d_half, m * n * sizeof(half_t), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start));


	//matrix_mul<THRESHOLD, CHECK_BLOCK, real_t>(device_a, device_b, m, n, k, device_d, alpha, beta, device_c);
	//matrix_mult_dmr<THRESHOLD, CHECK_BLOCK, real_t, half_t>(device_a, device_b, m, n, k, device_d, device_d_half, alpha, beta, device_c);

	matrix_mult_tensor_dmr<THRESHOLD, CHECK_BLOCK, real_t, half_t>(device_a, device_b, m, n, k, device_d, device_d_half, alpha, beta, device_c);

	

	cudaMemcpy(host_d, device_d, m * n * sizeof(real_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_d_half, device_d_half, m * n * sizeof(half_t), cudaMemcpyDeviceToHost);

	
	checkCudaErrors(cudaEventRecord(stop));
  	checkCudaErrors(cudaEventSynchronize(stop));


  	float milliseconds = 0;

  	checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

  	printf("Time: %f ms\n", milliseconds);
  	printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(m) *
                                                n * k * 2) /
                                               (milliseconds / 1000.)) /
                               1e12);
	

 //    std::cout << "FLOAT" << std::endl;
	// for (int i = 0; i < 10; i++) {
	// 	for (int j = 0; j < 10; j++) {
	// 		std::cout << host_d[i * m + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
	// std::cout << "HALF" << std::endl;
	// for (int i = 0; i < 10; i++) {
	// 	for (int j = 0; j < 10; j++) {
	// 		std::cout << host_d_half[i * m + j] << " ";
	// 	}
	// 	std::cout << std::endl;
	// }
    
	return 0;
}
