#include "device_functions.h"

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_fma(real_t *d_R0, real_t INPUT_A, real_t INPUT_B,
		real_t OUTPUT_R) {
	register real_t acc = OUTPUT_R;
	register real_t input_a = INPUT_A;
	register real_t input_b = INPUT_B;
	register real_t input_a_neg = -INPUT_A;
	register real_t input_b_neg = -INPUT_B;

#pragma unroll UNROLL_MAX
	for (register uint32_t count = 0; count < (OPS / 4); count++) {
		acc = fma_inline<USEFASTMATH>(input_a, input_b, acc);
		acc = fma_inline<USEFASTMATH>(input_a_neg, input_b, acc);
		acc = fma_inline<USEFASTMATH>(input_a, input_b_neg, acc);
		acc = fma_inline<USEFASTMATH>(input_a_neg, input_b_neg, acc);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_add(real_t *d_R0, real_t INPUT_A, real_t INPUT_B,
		real_t OUTPUT_R) {
	register real_t acc = OUTPUT_R;
	register real_t input_a = OUTPUT_R;
	register real_t input_a_neg = -OUTPUT_R;

#pragma unroll UNROLL_MAX
	for (register uint32_t count = 0; count < (OPS / 4); count++) {
		acc = add_inline<USEFASTMATH>(acc, input_a);
		acc = add_inline<USEFASTMATH>(acc, input_a_neg);
		acc = add_inline<USEFASTMATH>(acc, input_a_neg);
		acc = add_inline<USEFASTMATH>(acc, input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_mul(real_t *d_R0, real_t INPUT_A, real_t INPUT_B,
		real_t OUTPUT_R) {
	register real_t acc = OUTPUT_R;
	register real_t input_a = INPUT_A;
	register real_t input_a_inv = 1.0 / INPUT_A;

#pragma unroll UNROLL_MAX
	for (register uint32_t count = 0; count < (OPS / 4); count++) {
		acc = mul_inline<USEFASTMATH>(acc, input_a);
		acc = mul_inline<USEFASTMATH>(acc, input_a_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_a_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}


template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_pythagorean(real_t *d_R0, real_t INPUT_A, real_t INPUT_B,
		real_t OUTPUT_R) {
	register real_t acc = 0;
	register real_t input_a = INPUT_A;
	register real_t input_b = INPUT_B;

#pragma unroll UNROLL_MAX
	for (register uint32_t count = 0; count < (OPS / 4); count++) {
		acc += pythagorean_identity(input_a, input_b);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

