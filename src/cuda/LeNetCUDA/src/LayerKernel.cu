/**
 * Functions that work for all layers
 *
 */

#include "LayerKernel.h"
#include "cudaUtil.h"
#include <assert.h>

__device__ float J(float *theta, int n) {
	return 1;
}

__global__ void gradient_check(float *theta_plus, float *theta_minus,
		float *d_vector, float *gradient_diff, int size_n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i > size_n)
		return;

	theta_plus[i] = theta_plus[i] + EPSILON;
	theta_minus[i] = theta_minus[i] - EPSILON;

	float grad_approx = (J(theta_plus, size_n) - J(theta_minus, size_n)) / (2 * EPSILON);

	gradient_diff[i] = fabs(grad_approx - d_vector[i]);
}

void call_gradient_check(float *theta_plus, float *theta_minus, float *d_vector,
		int size_n) {
	dim3 blocks, threads;
	cuda_gridsize(&threads, &blocks, size_n);
	//malloc gradient difference array
	float *gradient_diff = nullptr;
	float *host_gradient_diff = (float*) malloc(sizeof(float) * size_n);
	cudaMalloc(&gradient_diff, sizeof(float) * size_n);

	//calc gradient difference
	gradient_check<<<blocks, threads>>>(theta_plus, theta_minus, d_vector, gradient_diff, size_n);
	CudaCheckError();


	cudaMemcpy(host_gradient_diff, gradient_diff, sizeof(float) * size_n, cudaMemcpyDeviceToHost);

	for(int i = 0; i < size_n; i++){
		assert(host_gradient_diff[i] > MAX_ERROR_ALLOWED  && "Pau no treino, gradient is bigger than 10e-5");
	}

	if(gradient_diff)
		cudaFree(gradient_diff);

	if(host_gradient_diff)
		free(host_gradient_diff);
}

