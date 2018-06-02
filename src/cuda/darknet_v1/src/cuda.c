int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include "assert.h"
#include <stdlib.h>
#include <time.h>

void check_framework_errors(cudaError_t error) {
	if (error == cudaSuccess) {
		return;
	}
	char errorDescription[250];
	snprintf(errorDescription, "CUDA Framework error: %s. Bailing.",
			cudaGetErrorString(error));
#ifdef LOGS
	log_error_detail((char *)errorDescription); end_log_file();
#endif
	printf("%s\n", errorDescription);
	exit(EXIT_FAILURE);
}

void* safe_cudaMalloc(size_t size) {
	void* devicePtr;
	void* goldPtr;
	void* outputPtr;
	printf("Passou aqui\n");
	// First, alloc DEVICE proposed memory and HOST memory for device memory checking
	check_framework_errors(cudaMalloc(&devicePtr, size));
	outputPtr = malloc(size);
	goldPtr = malloc(size);
	if ((outputPtr == NULL) || (goldPtr == NULL)) {
		log_error_detail((char *) "error host malloc");
		end_log_file();
		printf("error host malloc\n");
		exit(EXIT_FAILURE);
	}

	// ===> FIRST PHASE: CHECK SETTING BITS TO 10101010
	check_framework_errors(cudaMemset(devicePtr, 0xAA, size));
	memset(goldPtr, 0xAA, size);

	check_framework_errors(
			cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		check_framework_errors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END FIRST PHASE

	// ===> SECOND PHASE: CHECK SETTING BITS TO 01010101
	check_framework_errors(cudaMemset(devicePtr, 0x55, size));
	memset(goldPtr, 0x5, size);

	check_framework_errors(cudaMemcpy(outputPtr, devicePtr, size, cudaMemcpyDeviceToHost));
	if (memcmp(outputPtr, goldPtr, size)) {
		// Failed
		free(outputPtr);
		free(goldPtr);
		void* newDevicePtr = safe_cudaMalloc(size);
		check_framework_errors(cudaFree(devicePtr));
		return newDevicePtr;
	}
	// ===> END SECOND PHASE

	free(outputPtr);
	free(goldPtr);
	return devicePtr;
}

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
	dim3 d = {x, y, 1};
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
	static int init[16] = {0};
	static cublasHandle_t handle[16];
	int i = cuda_get_device();
	if (!init[i]) {
		cublasCreate(&handle[i]);
		init[i] = 1;
	}
	return handle[i];
}

float *cuda_make_array(float *x, size_t n) {
	float *x_gpu;
	size_t size = sizeof(float) * n;
#ifdef SAFE_MALLOC
	cudaError_t status;
	x_gpu = safe_cudaMalloc(size);
	if(x_gpu == 0)
		status = 0;
#else
	cudaError_t status = cudaMalloc((void **) &x_gpu, size);
#endif
	check_error(status);
	if (x) {
		status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
		check_error(status);
	}
	if (!x_gpu)
	error("Cuda malloc failed\n");
	return x_gpu;
}

void cuda_random(float *x_gpu, size_t n) {
	static curandGenerator_t gen[16];
	static int init[16] = {0};
	int i = cuda_get_device();
	if (!init[i]) {
		curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
		init[i] = 1;
	}
	curandGenerateUniform(gen[i], x_gpu, n);
	check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s) {
	float *tmp = calloc(n, sizeof(float));
	cuda_pull_array(x_gpu, tmp, n);
	//int i;
	//for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
	axpy_cpu(n, -1, x, 1, tmp, 1);
	float err = dot_cpu(n, tmp, 1, tmp, 1);
	printf("Error %s: %f\n", s, sqrt(err / n));
	free(tmp);
	return err;
}

int *cuda_make_int_array(size_t n) {
	int *x_gpu;
	size_t size = sizeof(int) * n;
	cudaError_t status = cudaMalloc((void **) &x_gpu, size);
	check_error(status);
	return x_gpu;
}

void cuda_free(float *x_gpu) {
	cudaError_t status = cudaFree(x_gpu);
	check_error(status);
}

void cuda_push_array(float *x_gpu, float *x, size_t n) {
	size_t size = sizeof(float) * n;
	cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
	check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n) {
	size_t size = sizeof(float) * n;
	cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
	check_error(status);
}

#endif
