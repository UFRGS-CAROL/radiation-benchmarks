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
	m = n = k = 2048;

	real_t alpha = 1;
	real_t beta = 1;

	real_t* host_a = (real_t*)calloc(m * k, sizeof(real_t));
	real_t* host_b = (real_t*)calloc(k * n, sizeof(real_t));
	real_t* host_c = (real_t*)calloc(m * n, sizeof(real_t));
	half_t* host_c_half = (half_t*)calloc(m * n, sizeof(half_t));

	for (int i = 0; i < m * k; i++) host_a[i] = alpha;
	for (int i = 0; i < m * k; i++) host_b[i] = beta;

	real_t *device_a, *device_b, *device_c;
	half_t *device_c_half;
	cudaMalloc((void**)&device_a, m * k * sizeof(real_t));
	cudaMalloc((void**)&device_b, k * n * sizeof(real_t));
	cudaMalloc((void**)&device_c, m * n * sizeof(real_t));
	cudaMalloc((void**)&device_c_half, m * n * sizeof(half_t));

	cudaMemcpy(device_a, host_a, m * k * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, k * n * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_c, host_c, m * n * sizeof(real_t), cudaMemcpyHostToDevice);
	cudaMemcpy(device_c_half, host_c_half, m * n * sizeof(half_t), cudaMemcpyHostToDevice);

	matrix_mult_dmr<THRESHOLD, CHECK_BLOCK>(device_a, device_b, m, n, k, device_c, device_c_half);

	cudaMemcpy(host_c, device_c, m * n * sizeof(real_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_c_half, device_c_half, m * n * sizeof(half_t), cudaMemcpyDeviceToHost);

    std::cout << "FLOAT" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_c[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "HALF" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << host_c_half[i * m + j] << " ";
		}
		std::cout << std::endl;
	}
    
	return 0;
}