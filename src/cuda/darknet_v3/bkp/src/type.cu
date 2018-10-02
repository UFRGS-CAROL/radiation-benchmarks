/*
 * type.c
 *
 *  Created on: 13/09/2018
 *      Author: fernando
 */

/**
 * Make real_t3 type
 * if cuda is activated it must be a function
 * accessible bye host or device
 */

#include "type.h"

#ifdef GPU
#include "cuda.h"
#endif
/**
 * Transform a float array into an half precision
 */

__global__ void float_to_half_array(real_t_device* dst, float* src,
		size_t size) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < size)
		dst[i] = __float2half(src[i]);
}

void transform_float_to_half_array(real_t_device* dst, float* src, size_t n) {
	float_to_half_array<<<cuda_gridsize(n), BLOCK>>>(dst, src, n);
	check_error(cudaPeekAtLastError());

}

/**
 * Read a file for all precisions
 */
int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp) {
	float* temp = (float*) calloc(times, sizeof(float));
	if (temp == NULL) {
		return -1;
	}
	int fread_result = fread(temp, sizeof(float), times, fp);
	if (fread_result != times) {
		free(temp);
		return -1;
	}
	int i;
	for (i = 0; i < times; i++) {
		//TODO: make ready for half
		dst[i] = real_t(temp[i]);
	}
	free(temp);
	return fread_result;

}

#if REAL_TYPE == HALF
FP16Array::FP16Array(size_t size, float* fp32_array) {
	cudaError_t status = cudaMalloc(&this->fp16_ptr,
			sizeof(real_t_fp16) * size);
	check_error(status);
	this->fp32_ptr = fp32_array;
	this->size = size;
}

FP16Array::~FP16Array() {
	if (this->fp16_ptr != nullptr) {
		cudaError_t status = cudaFree(this->fp16_ptr);
		check_error(status);
	}
}

__global__ void cuda_f32_to_f16(real_t_device* input_f32, size_t size,
		real_t_fp16 *output_f16) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		output_f16[idx] = __float2half(input_f32[idx]);
}

void FP16Array::cuda_convert_f32_to_f16() {
	cuda_f32_to_f16<<<this->size / BLOCK + 1, BLOCK>>>(this->fp32_ptr, this->size,
			this->fp16_ptr);
}

__global__ void cuda_f16_to_f32(real_t_fp16* input_f16, size_t size,
		float *output_f32) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		output_f32[idx] = __half2float(input_f16[idx]);
}

void FP16Array::cuda_convert_f16_to_f32() {
	cuda_f16_to_f32<<<this->size / BLOCK + 1, BLOCK>>>(this->fp16_ptr, this->size,
			this->fp32_ptr);
}

#endif
