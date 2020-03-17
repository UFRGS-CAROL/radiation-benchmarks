/*
 * micro_int_kernels.cu
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "Parameters.h"
#include "MicroInt.h"
#include "branch_kernel.h"
#include "input_device.h"

/**
 * dst is the output of the kernel
 * defined_src is defined input that has max threadIdx size
 */
template<uint32_t UNROLL_MAX, typename int_t>
__global__ void int_add_kernel(int_t* dst_1, int_t* dst_2, int_t* dst_3,
		const uint32_t op) {
	int_t acc = common_int_input[threadIdx.x];
	volatile int_t input_i = common_int_input[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < op; i++) {
		acc = acc + input_i;
		acc = acc - input_i;
		acc = acc + input_i;
		acc = acc - input_i;
	}

	dst_1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_3[blockIdx.x * blockDim.x + threadIdx.x] = acc;

}

template<uint32_t UNROLL_MAX>
__global__ void int_mul_kernel(int32_t* dst_1, int32_t* dst_2, int32_t* dst_3,
		uint32_t op) {
	volatile int32_t acc = common_int_input[threadIdx.x];
	volatile int32_t input_i = common_int_input[threadIdx.x];
	volatile int32_t divisor = inverse_mul_input[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < op; i++) {
		acc *= input_i;
		acc = __mulhi(acc, divisor);
		acc *= input_i;
		acc = __mulhi(acc, divisor);
	}

	dst_1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_3[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void int_mad_kernel(int_t* dst_1, int_t* dst_2, int_t* dst_3,
		uint32_t op) {
	int_t acc = common_int_input[threadIdx.x];
	volatile int_t input_i = common_int_input[threadIdx.x];
	volatile int_t input_i_neg = -input_i;

#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < op; i++) {
		acc += input_i * input_i;
		acc += input_i_neg * input_i;
		acc += input_i * input_i_neg;
		acc += input_i_neg * input_i_neg;
	}

	dst_1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
	dst_3[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<typename int_t>
void execute_kernel(MICROINSTRUCTION& micro, int_t* output_1, int_t* output_2,
		int_t* output_3, uint32_t grid_size, uint32_t block_size,
		uint32_t operation_num) {
	void (*kernel)(int_t*, int_t*, int_t*, uint32_t);
	switch (micro) {
	case ADD:
		kernel = int_add_kernel<LOOPING_UNROLL>;
		break;
	case MUL:
		kernel = int_mul_kernel<LOOPING_UNROLL>;
		break;
	case MAD:
	case FMA:
		kernel = int_mad_kernel<LOOPING_UNROLL>;
		break;
	case BRANCH:
		kernel = int_branch_kernel;
		break;
	}
	kernel<<<grid_size, block_size>>>(output_1, output_2, output_3,
			operation_num);
}

template<>
void MicroInt<int32_t>::execute_micro() {
	execute_kernel(this->parameters.micro, this->output_device_1.data(),
			this->output_device_2.data(), this->output_device_3.data(),
			this->grid_size, this->block_size, this->parameters.operation_num);
}

template<>
void MicroBranch<int32_t>::execute_micro() {
	execute_kernel(this->parameters.micro, this->output_device_1.data(),
			this->output_device_2.data(), this->output_device_3.data(),
			this->grid_size, this->block_size, this->parameters.operation_num);
}
