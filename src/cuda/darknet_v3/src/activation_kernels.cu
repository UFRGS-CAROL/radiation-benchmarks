#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "activations.h"
#include "cuda.h"
}

__device__ real_t lhtan_activate_kernel(real_t x) {
	if (x < 0)
		return .001f * x;
	if (x > 1)
		return .001f * (x - 1.f) + 1.f;
	return x;
}
__device__ real_t lhtan_gradient_kernel(real_t x) {
	if (x > 0 && x < 1)
		return 1;
	return .001;
}

__device__ real_t hardtan_activate_kernel(real_t x) {
	if (x < -1)
		return -1;
	if (x > 1)
		return 1;
	return x;
}
__device__ real_t linear_activate_kernel(real_t x) {
	return x;
}
__device__ real_t logistic_activate_kernel(real_t x) {
	return 1.f / (1.f + expf(-x));
}
__device__ real_t loggy_activate_kernel(real_t x) {
	return 2.f / (1.f + expf(-x)) - 1;
}
__device__ real_t relu_activate_kernel(real_t x) {
	return x * (x > 0);
}
__device__ real_t elu_activate_kernel(real_t x) {
	return (x >= 0) * x + (x < 0) * (expf(x) - 1);
}
__device__ real_t selu_activate_kernel(real_t x) {
	return (x >= 0) * 1.0507f * x + (x < 0) * 1.0507f * 1.6732f * (expf(x) - 1);
}
__device__ real_t relie_activate_kernel(real_t x) {
	return (x > 0) ? x : .01f * x;
}
__device__ real_t ramp_activate_kernel(real_t x) {
	return x * (x > 0) + .1f * x;
}
__device__ real_t leaky_activate_kernel(real_t x) {
	return (x > 0) ? x : .1f * x;
}
__device__ real_t tanh_activate_kernel(real_t x) {
	return (2.f / (1 + expf(-2 * x)) - 1);
}
__device__ real_t plse_activate_kernel(real_t x) {
	if (x < -4)
		return .01f * (x + 4);
	if (x > 4)
		return .01f * (x - 4) + 1;
	return .125f * x + .5f;
}
__device__ real_t stair_activate_kernel(real_t x) {
	int n = floorf(x);
	if (n % 2 == 0)
		return floorf(x / 2);
	else
		return (x - n) + floorf(x / 2);
}

__device__ real_t hardtan_gradient_kernel(real_t x) {
	if (x > -1 && x < 1)
		return 1;
	return 0;
}
__device__ real_t linear_gradient_kernel(real_t x) {
	return 1;
}
__device__ real_t logistic_gradient_kernel(real_t x) {
	return (1 - x) * x;
}
__device__ real_t loggy_gradient_kernel(real_t x) {
	real_t y = (x + 1) / 2;
	return 2 * (1 - y) * y;
}
__device__ real_t relu_gradient_kernel(real_t x) {
	return (x > 0);
}
__device__ real_t elu_gradient_kernel(real_t x) {
	return (x >= 0) + (x < 0) * (x + 1);
}
__device__ real_t selu_gradient_kernel(real_t x) {
	return (x >= 0) * 1.0507 + (x < 0) * (x + 1.0507 * 1.6732);
}
__device__ real_t relie_gradient_kernel(real_t x) {
	return (x > 0) ? 1 : .01f;
}
__device__ real_t ramp_gradient_kernel(real_t x) {
	return (x > 0) + .1f;
}
__device__ real_t leaky_gradient_kernel(real_t x) {
	return (x > 0) ? 1 : .1f;
}
__device__ real_t tanh_gradient_kernel(real_t x) {
	return 1 - x * x;
}
__device__ real_t plse_gradient_kernel(real_t x) {
	return (x < 0 || x > 1) ? .01f : .125f;
}
__device__ real_t stair_gradient_kernel(real_t x) {
	if (floorf(x) == x)
		return 0;
	return 1;
}

__device__ real_t activate_kernel(real_t x, ACTIVATION a) {
	switch (a) {
	case LINEAR:
		return linear_activate_kernel(x);
	case LOGISTIC:
		return logistic_activate_kernel(x);
	case LOGGY:
		return loggy_activate_kernel(x);
	case RELU:
		return relu_activate_kernel(x);
	case ELU:
		return elu_activate_kernel(x);
	case SELU:
		return selu_activate_kernel(x);
	case RELIE:
		return relie_activate_kernel(x);
	case RAMP:
		return ramp_activate_kernel(x);
	case LEAKY:
		return leaky_activate_kernel(x);
	case TANH:
		return tanh_activate_kernel(x);
	case PLSE:
		return plse_activate_kernel(x);
	case STAIR:
		return stair_activate_kernel(x);
	case HARDTAN:
		return hardtan_activate_kernel(x);
	case LHTAN:
		return lhtan_activate_kernel(x);
	}
	return 0;
}

__device__ real_t gradient_kernel(real_t x, ACTIVATION a) {
	switch (a) {
	case LINEAR:
		return linear_gradient_kernel(x);
	case LOGISTIC:
		return logistic_gradient_kernel(x);
	case LOGGY:
		return loggy_gradient_kernel(x);
	case RELU:
		return relu_gradient_kernel(x);
	case ELU:
		return elu_gradient_kernel(x);
	case SELU:
		return selu_gradient_kernel(x);
	case RELIE:
		return relie_gradient_kernel(x);
	case RAMP:
		return ramp_gradient_kernel(x);
	case LEAKY:
		return leaky_gradient_kernel(x);
	case TANH:
		return tanh_gradient_kernel(x);
	case PLSE:
		return plse_gradient_kernel(x);
	case STAIR:
		return stair_gradient_kernel(x);
	case HARDTAN:
		return hardtan_gradient_kernel(x);
	case LHTAN:
		return lhtan_gradient_kernel(x);
	}
	return 0;
}

__global__ void binary_gradient_array_kernel(real_t *x, real_t *dy, int n,
		int s, BINARY_ACTIVATION a, real_t *dx) {
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int i = id % s;
	int b = id / s;
	real_t x1 = x[b * s + i];
	real_t x2 = x[b * s + s / 2 + i];
	if (id < n) {
		real_t de = dy[id];
		dx[b * s + i] = x2 * de;
		dx[b * s + s / 2 + i] = x1 * de;
	}
}

extern "C" void binary_gradient_array_gpu(real_t *x, real_t *dx, int n,
		int size, BINARY_ACTIVATION a, real_t *y, cudaStream_t st) {
	binary_gradient_array_kernel<<<cuda_gridsize(n / 2), BLOCK, 0, st>>>(x, dx,
			n / 2, size, a, y);
	check_error(cudaPeekAtLastError());
}
__global__ void binary_activate_array_kernel(real_t *x, int n, int s,
		BINARY_ACTIVATION a, real_t *y) {
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int i = id % s;
	int b = id / s;
	real_t x1 = x[b * s + i];
	real_t x2 = x[b * s + s / 2 + i];
	if (id < n)
		y[id] = x1 * x2;
}

extern "C" void binary_activate_array_gpu(real_t *x, int n, int size,
		BINARY_ACTIVATION a, real_t *y, cudaStream_t st) {
	binary_activate_array_kernel<<<cuda_gridsize(n / 2), BLOCK, 0, st>>>(x,
			n / 2, size, a, y);
	check_error(cudaPeekAtLastError());
}

__global__ void activate_array_kernel(real_t *x, int n, ACTIVATION a) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		x[i] = activate_kernel(x[i], a);
}

__global__ void gradient_array_kernel(real_t *x, int n, ACTIVATION a,
		real_t *delta) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		delta[i] *= gradient_kernel(x[i], a);
}

extern "C" void activate_array_gpu(real_t *x, int n, ACTIVATION a,
		cudaStream_t st) {
	activate_array_kernel<<<cuda_gridsize(n), BLOCK, 0, st>>>(x, n, a);
	check_error(cudaPeekAtLastError());
}

extern "C" void gradient_array_gpu(real_t *x, int n, ACTIVATION a,
		real_t *delta, cudaStream_t st) {
	gradient_array_kernel<<<cuda_gridsize(n), BLOCK, 0, st>>>(x, n, a, delta);
	check_error(cudaPeekAtLastError());
}
