int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

void cuda_set_device(int n) {
	gpu_index = n;
	cudaError_t status = cudaSetDevice(n);
	check_error(status);
}

int cuda_get_device() {
	int n = 0;
	cudaError_t status = cudaGetDevice(&n);
	check_error(status);
	return n;
}

void check_error(cudaError_t status) {
	//cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error: %s", s);
		error(buffer);
	}
	if (status2 != cudaSuccess) {
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		printf("CUDA Error Prev: %s\n", s);
		assert(0);
		snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		error(buffer);
	}
}

dim3 cuda_gridsize(size_t n) {
	size_t k = (n - 1) / BLOCK + 1;
	size_t x = k;
	size_t y = 1;
	if (x > 65535) {
		x = ceil(sqrt(k));
		y = (n - 1) / (x * BLOCK) + 1;
	}
	dim3 d = { x, y, 1 };
	//printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
	return d;
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
	static int init[16] = {0};
	static cudnnHandle_t handle[16];
	int i = cuda_get_device();
	if(!init[i]) {
		cudnnCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}
#endif

cublasHandle_t blas_handle() {
	static int init[16] = { 0 };
	static cublasHandle_t handle[16];
	int i = cuda_get_device();
	if (!init[i]) {
		cublasCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}

real_t_device *cuda_make_array(real_t *x, size_t n) {
	real_t_device *x_gpu;
	size_t size = sizeof(real_t_device) * n;
	cudaError_t status = cudaMalloc((void **) &x_gpu, size);
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	} else {
		fill_gpu(n, 0, x_gpu, 1);
	}
	if (!x_gpu)
		error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_random(real_t_device *x_gpu, size_t n) {
	static curandGenerator_t gen[16];
	static int init[16] = { 0 };
	int i = cuda_get_device();
	if (!init[i]) {
		curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
		init[i] = 1;
	}

#if REAL_TYPE == HALF
	float* tmp_d;
	cudaError_t status = cudaMalloc((void **) &tmp_d, sizeof(real_t) * n);
	check_error(status);

	curandStatus_t stss = curandGenerateUniform(gen[i], tmp_d, n);

	transform_float_to_half_array(x_gpu, tmp_d, n);
	cudaFree(tmp_d);

#elif REAL_TYPE == FLOAT
	curandStatus_t status = curandGenerateUniform(gen[i], x_gpu, n);
#elif REAL_TYPE == DOUBLE
	curandStatus_t status = curandGenerateUniformDouble(gen[i], x_gpu, n);
#endif

	check_error(cudaPeekAtLastError());
}

real_t cuda_compare(real_t_device *x_gpu, real_t *x, size_t n, char *s) {
	real_t *tmp = (real_t*) calloc(n, sizeof(real_t));
	cuda_pull_array(x_gpu, tmp, n);
	//int i;
	//for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
	axpy_cpu(n, real_t(-1), x, 1, tmp, 1);
	real_t err = dot_cpu(n, tmp, 1, tmp, 1);
	printf("Error %s: %f\n", s, sqrt(err / n));
	free(tmp);
	return err;
}

int *cuda_make_int_array(int *x, size_t n) {
	int *x_gpu;
	size_t size = sizeof(int) * n;
	cudaError_t status = cudaMalloc((void **) &x_gpu, size);
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	}
	if (!x_gpu)
		error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_free(real_t_device *x_gpu) {
	cudaError_t status = cudaFree(x_gpu);
	check_error(status);
}

void cuda_push_array(real_t_device *x_gpu, real_t *x, size_t n) {
	size_t size = sizeof(real_t) * n;
	cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
	check_error(status);
}

void cuda_pull_array(real_t_device *x_gpu, real_t *x, size_t n) {
	size_t size = sizeof(real_t) * n;
	cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
	check_error(status);
}

real_t cuda_mag_array(real_t_device *x_gpu, size_t n) {
	real_t *temp = (real_t*) calloc(n, sizeof(real_t));
	cuda_pull_array(x_gpu, temp, n);
	real_t m = mag_array(temp, n);
	free(temp);
	return m;
}
#else
void cuda_set_device(int n) {
}

#endif
