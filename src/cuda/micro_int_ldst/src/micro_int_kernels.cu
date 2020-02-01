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

void MicroInt::select_micro() {
	switch (this->micro) {
	case ADD_INT:
		add_int_kernel<<<this->grid, this->block>>>(this->input_device.data(),
				this->output_device.data());
	}
}
