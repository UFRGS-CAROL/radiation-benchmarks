/*
 * micro_int_kernels.cu
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "Parameters.h"
#include "MicroInt.h"
//#include "ldst_kernel.h"
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
		output = output + input;
		output = output + output;
		output = output - input;
		output = output + input;
		output = output - input - input - input;
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

template<uint32_t UNROLL_MAX, typename int_t> __forceinline__
__device__ void ldst_same_direction_kernel(int_t *dst, int_t *src) {
#pragma unroll UNROLL_MAX
	for(uint32_t i = 0; i < UNROLL_MAX; i++){
		dst[i] = src[i];
	}
}

template<uint32_t MAX_MOVEMENTS, typename int_t>
__global__ void ldst_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) * op;
	int_t* dst_ptr = dst + thread_id;
	int_t* src_ptr = src + thread_id;

#pragma unroll MAX_MOVEMENTS
	for(uint32_t i = 0; i < MAX_MOVEMENTS; i++){
		//copy to dst
		ldst_same_direction_kernel<MEM_OPERATION_NUM>(dst_ptr, src_ptr);
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
		kernel = ldst_int_kernel<MAX_THREAD_LD_ST_OPERATIONS>;
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
__global__ void check_kernel(int_t* lhs, int_t* rhs) {
	const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (lhs[thread_id] != rhs[thread_id]) {
		atomicAdd(&errors, 1);
	}
}

template<typename int_t>
size_t call_checker(int_t* lhs, int_t* rhs, size_t array_size,
		MICROINSTRUCTION& micro) {

	size_t grid = array_size / (MAX_THREAD_BLOCK);
	check_kernel<<<grid, MAX_THREAD_BLOCK>>>(lhs, rhs);

	unsigned long long herrors = 0;
	rad::checkFrameworkErrors(cudaPeekAtLastError());
	rad::checkFrameworkErrors(cudaDeviceSynchronize());
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol(&herrors, errors, sizeof(unsigned long long)));
	return herrors;
}

template<>
size_t MicroInt<int32_t>::compare_on_gpu() {
	return call_checker(this->output_device.data(), this->input_device.data(),
			this->array_size, this->parameters.micro);
}
