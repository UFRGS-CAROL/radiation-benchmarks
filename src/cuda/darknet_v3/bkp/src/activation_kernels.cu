#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

//extern "C" {
#include "activations.h"
#include "cuda.h"
//}

__device__ real_t_device lhtan_activate_kernel(real_t_device x) {
	if (x < real_t_device(0.0))
		return real_t_device(.001f) * x;
	if (x > real_t_device(1.0))
		return real_t_device(.001f) * (x - real_t_device(1.f))
				+ real_t_device(1.f);
	return x;
}
__device__ real_t_device lhtan_gradient_kernel(real_t_device x) {
	if (x > real_t_device(0) && x < real_t_device(1))
		return real_t_device(1);
	return real_t_device(.001);
}

__device__ real_t_device hardtan_activate_kernel(real_t_device x) {
	if (x < real_t_device(-1))
		return -1;
	if (x > real_t_device(1))
		return real_t_device(1);
	return x;
}
__device__ real_t_device linear_activate_kernel(real_t_device x) {
	return x;
}
__device__ real_t_device logistic_activate_kernel(real_t_device x) {
	return real_t_device(1.f) / (real_t_device(1.f) + exp_real(-x));
}
__device__ real_t_device loggy_activate_kernel(real_t_device x) {
	return real_t_device(2.f) / (real_t_device(1.f) + exp_real(-x)) - real_t_device(1);
}
__device__ real_t_device relu_activate_kernel(real_t_device x) {
	return x * real_t_device(x > real_t_device(0));
}
__device__ real_t_device elu_activate_kernel(real_t_device x) {
	return real_t_device(x >= real_t_device(0)) * x
			+ real_t_device(x < real_t_device(0))
					* (exp_real(x) - real_t_device(1));
}
__device__ real_t_device selu_activate_kernel(real_t_device x) {
	return real_t_device(x >= real_t_device(0)) * real_t_device(1.0507f) * x
			+ real_t_device(x < real_t_device(0)) * real_t_device(1.0507f)
					* real_t_device(1.6732f) * (exp_real(x) - real_t_device(1));
}
__device__ real_t_device relie_activate_kernel(real_t_device x) {
	return (x > real_t_device(0)) ? x : real_t_device(.01f) * x;
}
__device__ real_t_device ramp_activate_kernel(real_t_device x) {
	return x * real_t_device(x > real_t_device(0)) + real_t_device(.1f) * x;
}
__device__ real_t_device leaky_activate_kernel(real_t_device x) {
	return (x > real_t_device(0)) ? x : real_t_device(.1f) * x;
}
__device__ real_t_device tanh_activate_kernel(real_t_device x) {
	return (real_t_device(2.f)
			/ (real_t_device(1) + exp_real(real_t_device(-2) * x))
			- real_t_device(1));
}
__device__ real_t_device plse_activate_kernel(real_t_device x) {
	if (x < real_t_device(-4))
		return real_t_device(.01f) * (x + real_t_device(4));
	if (x > real_t_device(4))
		return real_t_device(.01f) * (x - real_t_device(4)) + real_t_device(1);
	return real_t_device(.125f) * x + real_t_device(.5f);
}
__device__ real_t_device stair_activate_kernel(real_t_device x) {
	int n = floor_real(x);
	if (n % 2 == 0)
		return floor_real(x / real_t_device(2));
	else
		return (x - real_t_device(n)) + floor_real(x / real_t_device(2));
}

__device__ real_t_device hardtan_gradient_kernel(real_t_device x) {
	if (x > real_t_device(-1) && x < real_t_device(1))
		return real_t_device(1);
	return real_t_device(0);
}
__device__ real_t_device linear_gradient_kernel(real_t_device x) {
	return real_t_device(1);
}
__device__ real_t_device logistic_gradient_kernel(real_t_device x) {
	return (real_t_device(1) - x) * x;
}
__device__ real_t_device loggy_gradient_kernel(real_t_device x) {
	real_t_device y = (x + real_t_device(1)) / real_t_device(2);
	return real_t_device(2) * (real_t_device(1) - y) * y;
}
__device__ real_t_device relu_gradient_kernel(real_t_device x) {
	return (x > real_t_device(0));
}
__device__ real_t_device elu_gradient_kernel(real_t_device x) {
	return real_t_device(x >= real_t_device(0)) + real_t_device(x < real_t_device(0)) * (x + real_t_device(1));
}
__device__ real_t_device selu_gradient_kernel(real_t_device x) {
	return real_t_device(x >= real_t_device(0)) * real_t_device(1.0507)
			+ real_t_device(x < real_t_device(0))
					* (x + real_t_device(1.0507) * real_t_device(1.6732));
}
__device__ real_t_device relie_gradient_kernel(real_t_device x) {
	return (x > real_t_device(0)) ? real_t_device(1) : real_t_device(.01f);
}
__device__ real_t_device ramp_gradient_kernel(real_t_device x) {
	return real_t_device(x > real_t_device(0)) + real_t_device(.1f);
}
__device__ real_t_device leaky_gradient_kernel(real_t_device x) {
	return real_t_device(x > real_t_device(0)) ? real_t_device(1) : real_t_device(.1f);
}
__device__ real_t_device tanh_gradient_kernel(real_t_device x) {
	return real_t_device(1) - x * x;
}
__device__ real_t_device plse_gradient_kernel(real_t_device x) {
	return real_t_device(x < real_t_device(0) || x > real_t_device(1)) ? real_t_device(.01f) : real_t_device(.125f);
}
__device__ real_t_device stair_gradient_kernel(real_t_device x) {
	if (floor_real(x) == x)
		return real_t_device(0);
	return real_t_device(1);
}

__device__ real_t_device activate_kernel(real_t_device x, ACTIVATION a) {
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

__device__ real_t_device gradient_kernel(real_t_device x, ACTIVATION a) {
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
	return real_t_device(0);
}

__global__ void binary_gradient_array_kernel(real_t_device *x,
		real_t_device *dy, int n, int s, BINARY_ACTIVATION a,
		real_t_device *dx) {
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int i = id % s;
	int b = id / s;
	real_t_device x1 = x[b * s + i];
	real_t_device x2 = x[b * s + s / 2 + i];
	if (id < n) {
		real_t_device de = dy[id];
		dx[b * s + i] = x2 * de;
		dx[b * s + s / 2 + i] = x1 * de;
	}
}

//extern "C"

void binary_gradient_array_gpu(real_t_device *x, real_t_device *dx, int n,
		int size, BINARY_ACTIVATION a, real_t_device *y) {
	binary_gradient_array_kernel<<<cuda_gridsize(n / 2), BLOCK>>>(x, dx, n / 2,
			size, a, y);
	check_error(cudaPeekAtLastError());
}
__global__ void binary_activate_array_kernel(real_t_device *x, int n, int s,
		BINARY_ACTIVATION a, real_t_device *y) {
	int id = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	int i = id % s;
	int b = id / s;
	real_t_device x1 = x[b * s + i];
	real_t_device x2 = x[b * s + s / 2 + i];
	if (id < n)
		y[id] = x1 * x2;
}

//extern "C"
void binary_activate_array_gpu(real_t_device *x, int n, int size,
		BINARY_ACTIVATION a, real_t_device *y) {
	binary_activate_array_kernel<<<cuda_gridsize(n / 2), BLOCK>>>(x, n / 2,
			size, a, y);
	check_error(cudaPeekAtLastError());
}

__global__ void activate_array_kernel(real_t_device *x, int n, ACTIVATION a) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		x[i] = activate_kernel(x[i], a);
}

__global__ void gradient_array_kernel(real_t_device *x, int n, ACTIVATION a,
		real_t_device *delta) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n)
		delta[i] *= gradient_kernel(x[i], a);
}

//extern "C"
void activate_array_gpu(real_t_device *x, int n, ACTIVATION a) {
	activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
	check_error(cudaPeekAtLastError());
}

//extern "C"
void gradient_array_gpu(real_t_device *x, int n, ACTIVATION a,
		real_t_device *delta) {
	gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
	check_error(cudaPeekAtLastError());
}
