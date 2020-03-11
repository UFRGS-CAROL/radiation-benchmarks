/*
 * micro_int_kernels.cu
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "Parameters.h"
#include "MicroInt.h"
#include "branch_kernel.h"

/**
 * dst is the output of the kernel
 * defined_src is defined input that has max threadIdx size
 */
template<uint32_t UNROLL_MAX, typename int_t>
__global__ void add_int_kernel(int_t* src, int_t* dst, const uint32_t op) {
	volatile register int_t acc = src[threadIdx.x];
	volatile register int_t input_i = src[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < op; i++) {
		acc = acc + input_i;
		acc = acc - input_i;
		acc = acc + input_i;
		acc = acc - input_i;
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void mul_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	int_t acc = src[threadIdx.x];
	volatile int_t input_i = src[threadIdx.x];
	volatile int_t divider = input_i * input_i; //^2
	divider *= divider; //^4
	divider *= divider; //^8

#pragma unroll 32
	for (uint32_t i = 0; i < op; i++) {

#pragma unroll 8
		for (uint32_t k = 0; k < 8; k++)
			acc *= input_i;

		acc = acc / divider;
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, typename int_t>
__global__ void mad_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	int_t acc = src[threadIdx.x];
	volatile int_t input_i = src[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < op; i++) {
		acc += input_i * input_i;
		acc -= input_i * input_i;
		acc -= input_i * input_i;
		acc += input_i * input_i;
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, typename int_t> __forceinline__
__device__ void ldst_same_direction_kernel(int_t *dst, int_t *src) {
#pragma unroll UNROLL_MAX
	for (uint32_t i = 0; i < UNROLL_MAX; i++) {
		dst[i] = src[i];
	}
}

template<uint32_t MAX_MOVEMENTS, typename int_t>
__global__ void ldst_int_kernel(int_t* src, int_t* dst, uint32_t op) {
	const uint32_t thread_id = (blockIdx.x * blockDim.x + threadIdx.x) * op;
	int_t* dst_ptr = dst + thread_id;
	int_t* src_ptr = src + thread_id;

#pragma unroll MAX_MOVEMENTS
	for (uint32_t i = 0; i < MAX_MOVEMENTS; i++) {
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
		kernel = add_int_kernel<LOOPING_UNROLL>;
		break;
	case MUL_INT:
		kernel = mul_int_kernel<LOOPING_UNROLL>;
		break;
	case MAD_INT:
		kernel = mad_int_kernel<LOOPING_UNROLL>;
		break;
	case LDST:
		kernel = ldst_int_kernel<MAX_THREAD_LD_ST_OPERATIONS>;
		break;
	case BRANCH:
		kernel = branch_int_kernel<MAX_THREAD_LD_ST_OPERATIONS>;
		break;
	}
	kernel<<<grid_size, block_size>>>(input, output, operation_num);
}

template<>
void MicroInt<int32_t>::execute_micro() {
	execute_kernel(this->parameters.micro, this->input_device.data(),
			this->output_device.data(), this->grid_size, this->block_size,
			this->parameters.operation_num);
}

/**
 *
 * May be useful in the future
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
 rad::checkFrameworkErrors(cudaGetLastError());
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

 */
