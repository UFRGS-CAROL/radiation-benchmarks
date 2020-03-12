#include "Micro.h"
#include "device_functions.h"

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_fma(real_t *dst, real_t *src, const uint32_t ops) {
	real_t acc = src[threadIdx.x];
	real_t input_i = src[threadIdx.x];
	real_t input_i_neg = -input_i;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = fma_inline<USEFASTMATH>(input_i, input_i, acc);
		acc = fma_inline<USEFASTMATH>(input_i_neg, input_i, acc);
		acc = fma_inline<USEFASTMATH>(input_i, input_i_neg, acc);
		acc = fma_inline<USEFASTMATH>(input_i_neg, input_i_neg, acc);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_add(real_t *dst, real_t *src, const uint32_t ops) {
	real_t acc = src[threadIdx.x];
	real_t input_i = src[threadIdx.x];
	real_t input_i_neg = -input_i;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = add_inline<USEFASTMATH>(acc, input_i);
		acc = add_inline<USEFASTMATH>(acc, input_i_neg);
		acc = add_inline<USEFASTMATH>(acc, input_i);
		acc = add_inline<USEFASTMATH>(acc, input_i_neg);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_mul(real_t *dst, real_t *src, const uint32_t ops) {
	real_t acc = src[threadIdx.x];
	real_t input_i = src[threadIdx.x];
	real_t input_i_inv = real_t(1.0f) / input_i;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = mul_inline<USEFASTMATH>(acc, input_i);
		acc = mul_inline<USEFASTMATH>(acc, input_i_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_i_inv);
		acc = mul_inline<USEFASTMATH>(acc, input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_div(real_t *dst, real_t *src, const uint32_t ops) {
	real_t acc = src[threadIdx.x];
	real_t input_i = src[threadIdx.x];
	real_t input_i_inv = real_t(1) / input_i;

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc = div_inline<USEFASTMATH>(acc, input_i_inv);
		acc = div_inline<USEFASTMATH>(acc, input_i);
		acc = div_inline<USEFASTMATH>(acc, input_i_inv);
		acc = div_inline<USEFASTMATH>(acc, input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_pythagorean(real_t *dst, real_t *src,
		const uint32_t ops) {
	real_t acc = 0;
	real_t input_i = src[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += pythagorean_identity<USEFASTMATH>(input_i, input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void micro_kernel_euler(real_t *dst, real_t *src,
		const uint32_t ops) {
	real_t acc = 0;
	real_t input_i = src[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += euler<USEFASTMATH>(-input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<typename real_t>
void execute_kernel(MICROINSTRUCTION& micro, real_t* output, real_t* input,
		size_t grid_size, size_t block_size, uint32_t operation_num,
		bool fast_math) {

	//	real_t, real_t,
	void (*kernel)(real_t*, real_t*, uint32_t);
	switch (micro) {
	case ADD:
		kernel = micro_kernel_add<LOOPING_UNROLL, true>;
		break;
	case MUL:
		kernel = micro_kernel_mul<LOOPING_UNROLL, true>;
		break;
	case FMA:
		kernel = micro_kernel_fma<LOOPING_UNROLL, true>;
		break;
	case DIV:
		if (fast_math) {
			kernel = micro_kernel_div<LOOPING_UNROLL, true>;
		} else {
			kernel = micro_kernel_div<LOOPING_UNROLL, false>;
		}

		break;
	case PYTHAGOREAN:
		if (fast_math) {
			kernel = micro_kernel_pythagorean<LOOPING_UNROLL, true>;

		} else {
			kernel = micro_kernel_pythagorean<LOOPING_UNROLL, false>;

		}
		break;
	case EULER:
		if (fast_math) {
			kernel = micro_kernel_euler<LOOPING_UNROLL, true>;

		} else {
			kernel = micro_kernel_euler<LOOPING_UNROLL, false>;

		}
		break;
	}

	kernel<<<grid_size, block_size>>>(output, input, operation_num);
}

template<>
void Micro<float>::execute_micro() {
	execute_kernel(this->parameters.micro, this->output_device.data(),
			this->input_device.data(), this->parameters.grid_size,
			this->parameters.block_size, this->parameters.operation_num,
			this->parameters.fast_math);

}
