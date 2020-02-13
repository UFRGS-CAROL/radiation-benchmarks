#include "Micro.h"
#include "device_functions.h"

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_fma(real_t *d_R0, real_t input_a, real_t input_b,
		real_t output_r, const uint32_t ops) {
	real_t acc = output_r;
	real_t input_a_neg = -input_a;
	real_t input_b_neg = -input_b;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = fma_inline<USEFASTMATH>(input_a, input_b, acc);
		acc = fma_inline<USEFASTMATH>(input_a_neg, input_b, acc);
		acc = fma_inline<USEFASTMATH>(input_a, input_b_neg, acc);
		acc = fma_inline<USEFASTMATH>(input_a_neg, input_b_neg, acc);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_add(real_t *d_R0, real_t input_a, real_t input_b,
		real_t output_r, const uint32_t ops) {
	real_t acc = output_r;
	real_t input_a_neg = -output_r;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = add_inline<USEFASTMATH>(acc, input_a);
		acc = add_inline<USEFASTMATH>(acc, input_a_neg);
		acc = add_inline<USEFASTMATH>(acc, input_a_neg);
		acc = add_inline<USEFASTMATH>(acc, input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_mul(real_t *d_R0, real_t input_a, real_t input_b,
		real_t output_r, const uint32_t ops) {
	real_t acc = output_r;
	real_t input_a_inv = real_t(1.0) / input_a;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = mul_inline<USEFASTMATH>(acc, input_a);
		acc = mul_inline<USEFASTMATH>(acc, input_a_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_a_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_pythagorean(real_t *d_R0, real_t input_a,
		real_t input_b, real_t output_r, const uint32_t ops) {
	real_t acc = output_r;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += pythagorean_identity<USEFASTMATH>(input_a, input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_euler(real_t *d_R0, real_t input_a, real_t input_b,
		real_t output_r, const uint32_t ops) {
	real_t acc = 0;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += euler<USEFASTMATH>(input_a);
	}

	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<bool USEFASTMATH, typename real_t>
void execute_kernel(MICROINSTRUCTION& micro, real_t* output, real_t input_a,
		real_t input_b, real_t output_acc, size_t grid_size, size_t block_size,
		uint32_t operation_num) {

	void (*kernel)(real_t*, real_t, real_t, real_t, uint32_t);
	switch (micro) {
	case ADD:
//		kernel = micro_kernel_add<LOOPING_UNROLL, USEFASTMATH>;
		break;
	case MUL:
//		kernel = micro_kernel_mul<LOOPING_UNROLL, USEFASTMATH>;
		break;
	case FMA:
//		kernel = micro_kernel_fma<LOOPING_UNROLL, USEFASTMATH>;
		break;
		throw_line("Not implemented yet")
		;
		break;
	case PYTHAGOREAN:
		kernel = micro_kernel_pythagorean<LOOPING_UNROLL, USEFASTMATH>;
		break;
	case EULER:
		kernel = micro_kernel_euler<LOOPING_UNROLL, USEFASTMATH>;
		break;
	}
	kernel<<<grid_size, block_size>>>(output, input_a, input_b, output_acc,
			operation_num);
}

template<>
void Micro<float>::execute_micro() {
	if (this->parameters.fast_math) {
		execute_kernel<true>(this->parameters.micro, this->output_device.data(),
				this->input_kernel.INPUT_A, this->input_kernel.INPUT_B,
				this->input_kernel.OUTPUT_R, this->parameters.grid_size,
				this->parameters.block_size, this->parameters.operation_num);
	} else {
		execute_kernel<false>(this->parameters.micro,
				this->output_device.data(), this->input_kernel.INPUT_A,
				this->input_kernel.INPUT_B, this->input_kernel.OUTPUT_R,
				this->parameters.grid_size, this->parameters.block_size,
				this->parameters.operation_num);
	}
}
