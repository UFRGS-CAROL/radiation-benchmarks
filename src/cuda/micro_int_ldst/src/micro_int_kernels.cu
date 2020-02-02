/*
 * micro_int_kernels.cu
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "Parameters.h"
#include "MicroInt.h"

/**
 * dst is the output of the kernel
 * defined_src is defined input that has max threadIdx size
 */
__global__ void add_int_kernel(int32_t* src, int32_t* dst, uint32_t op) {

	int32_t output = src[threadIdx.x];
	int32_t input = src[threadIdx.x];
#pragma unroll
	for (uint32_t i = 0; i < op; i++) {
		output += input + output;
		output -= input;
		output += input;
		output -= input - output;
	}

	const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	dst[thread_id] = output;
}

__global__ void mul_int_kernel(int32_t* defined_src, int32_t* dst,
		uint32_t op) {

	int32_t output_register = defined_src[threadIdx.x];
	int32_t input_register = defined_src[threadIdx.x];
#pragma unroll
	for (uint32_t i = 0; i < op; i++) {
		output_register *= (input_register * 2);
		output_register /= input_register;
		output_register *= input_register;
		output_register /= (input_register * 2);
	}

	const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	dst[thread_id] = output_register;
}

__global__ void mad_int_kernel(int32_t* src, int32_t* dst, uint32_t op) {
	int32_t output = src[threadIdx.x];
	int32_t input = src[threadIdx.x];
#pragma unroll
	for (uint32_t i = 0; i < op; i++) {
		output += output * input * 2;
		output -= output * input;
		output += output * input;
		output -= output * input * 2;
	}

	const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	dst[thread_id] = output;
}

__global__ void ldst_int_kernel(int32_t* src, int32_t* dst, uint32_t op) {
	const uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) * op;
	for (uint32_t i = 0; i < op; i++) {
		dst[i + thread_id] = src[i + thread_id];
	}
}

void MicroInt::execute_micro() {
	void (*kernel)(int32_t*, int32_t*, uint32_t);
	switch (this->parameters.micro) {
	case ADD_INT:
		kernel = add_int_kernel;
		break;
	case MUL_INT:
		kernel = mul_int_kernel;
		break;
	case MAD_INT:
		kernel = mad_int_kernel;
		break;
	case LDST:
		kernel = ldst_int_kernel;
		break;
	}

	kernel<<<this->parameters.grid_size, this->parameters.block_size>>>(
			this->input_device.data(), this->output_device.data(), this->parameters.operation_num);
}
