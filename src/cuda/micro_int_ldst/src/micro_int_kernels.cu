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
template<typename int_t>
__global__ void add_int_kernel(int_t* src, int_t* dst, uint32_t op) {

	int_t output = src[threadIdx.x];
	int_t input = src[threadIdx.x];
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

template<typename int_t>
__global__ void mul_int_kernel(int_t* defined_src, int_t* dst, uint32_t op) {

	int_t output_register = defined_src[threadIdx.x];
	int_t input_register = defined_src[threadIdx.x];
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

template<typename int_t>
__global__ void mad_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	int_t output = src[threadIdx.x];
	int_t input = src[threadIdx.x];
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

template<typename int_t>
__global__ void ldst_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) * op;
	if(thread_id != 0)
	for (uint32_t i = thread_id; i < thread_id + op; i++) {
		dst[i] = src[i];
	}
}

template<typename int_t>
__global__ void check_kernel(int_t* lhs, int_t* rhs) {
	const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ unsigned long long shared_errors;
	if(threadIdx.x == 0){
		shared_errors = 0;
	}

	__syncthreads();

	if (lhs[thread_id] != rhs[thread_id]) {
		atomicAdd(&shared_errors, 1);
	}

	__syncthreads();
	if (threadIdx.x == 0 && shared_errors != 0) {
		atomicAdd(&errors, shared_errors);
	}
}


template<typename int_t>
void execute_kernel(MICROINSTRUCTION& micro, int_t* input, int_t* output,
		uint32_t grid_size, uint32_t block_size, uint32_t operation_num) {
	void (*kernel)(int_t*, int_t*, uint32_t);
	switch (micro) {
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

	kernel<<<grid_size, block_size>>>(input, output, operation_num);
}

template<>
void MicroInt<int32_t>::execute_micro() {
	execute_kernel(this->parameters.micro, this->input_device.data(),
			this->output_device.data(), this->grid_size, this->block_size,
			this->operation_num);
}



template<typename int_t>
void call_checker(int_t* lhs, int_t* rhs, size_t array_size) {

	size_t grid = array_size / MAX_THREAD_BLOCK;
	check_kernel<<<grid, MAX_THREAD_BLOCK>>>(lhs, rhs);
	rad::checkFrameworkErrors(cudaPeekAtLastError());
	rad::checkFrameworkErrors(cudaDeviceSynchronize());
}

template<>
void MicroInt<int32_t>::compare_on_gpu() {
	call_checker(this->output_device.data(), this->input_device.data(),
			this->array_size);
}
