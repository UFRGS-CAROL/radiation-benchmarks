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
__global__ void add_int_kernel(int32_t* defined_src, int32_t* dst) {

	volatile int32_t output_register = defined_src[threadIdx.x];
	volatile int32_t input_register = defined_src[threadIdx.x];
#pragma unroll
	for (uint32_t i = 0; i < OPS; i++) {
		dst += input_register;
		dst -= input_register;
		dst += input_register;
		dst -= input_register;
	}

	uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	dst[thread_id] = output_register;
}

__global__ void mul_int_kernel(int32_t* defined_src, int32_t* dst) {
}

__global__ void mad_int_kernel(int32_t* defined_src, int32_t* dst) {
}

__global__ void ldst_int_kernel(int32_t* defined_src, int32_t* dst) {
}

void MicroInt::execute_micro() {
	void (*kernel)(int32_t*, int32_t*);
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

	kernel<<<this->parameters.grid_size, this->parameters.block_size>>>(this->input_device.data(),
					this->output_device.data());
}
