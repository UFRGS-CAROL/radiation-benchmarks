#include "kernels.h"
#include <cassert>
#include <vector>
#include <iostream>

#define CHECK_BLOCK 16
#define THRESHOLD 1



int main(int argc, char **argv) {

	int m;
	int n;
	int k;
	m = n = k = 4096;

	float alpha = 1;
	float beta = 1;

	float* host_a = (float*)calloc(m * k, sizeof(float));
	float* host_b = (float*)calloc(k * n, sizeof(float));
	float* host_c = (float*)calloc(m * n, sizeof(float));
	float* host_d = (float*)calloc(m * n, sizeof(float));
	float* host_d_half = (float*)calloc(m * n, sizeof(float));

	for (int i = 0; i < m * k; i++) host_a[i] = alpha;
	for (int i = 0; i < m * k; i++) host_b[i] = beta;
	for (int i = 0; i < m * k; i++) host_c[i] = 0;	

	float *device_a, *device_b, *device_c, *device_d;
	float *device_d_half;
	cudaMalloc((void**)&device_a, m * k * sizeof(float));
	cudaMalloc((void**)&device_b, k * n * sizeof(float));
	cudaMalloc((void**)&device_c, m * n * sizeof(float));
	cudaMalloc((void**)&device_d, m * n * sizeof(float));
	cudaMalloc((void**)&device_d_half, m * n * sizeof(float));

	cudaMemcpy(device_a, host_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, k * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_c, host_c, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_d, host_d, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_d_half, host_d_half, m * n * sizeof(float), cudaMemcpyHostToDevice);

	matrix_mult_dmr<THRESHOLD, CHECK_BLOCK, float, float>(device_a, device_b, m, n, k, device_d, device_d_half, alpha, beta, device_c);

	
	cudaMemcpy(host_d, device_d, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_d_half, device_d_half, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "FLOAT" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_d[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "HALF" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_d_half[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
    
	return 0;
}