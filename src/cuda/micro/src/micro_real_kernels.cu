#include "MicroReal.h"
#include "device_functions.h"
#include "input_device.h"

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void real_fma_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = common_float_input[threadIdx.x];
	real_t input_i = common_float_input[threadIdx.x];
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
__global__ void real_add_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = common_float_input[threadIdx.x];
	real_t input_i = common_float_input[threadIdx.x];
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
__global__ void real_mul_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = common_float_input[threadIdx.x];
	real_t input_i = common_float_input[threadIdx.x];
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
__global__ void real_div_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = common_float_input[threadIdx.x];
	real_t input_i = common_float_input[threadIdx.x];
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
__global__ void real_pythagorean_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = 0;
	//convert to radians first
	real_t input_i = fabs(common_float_input[threadIdx.x]) * M_PI
			/ real_t(180.0f);

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += pythagorean_identity<USEFASTMATH>(input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<uint32_t UNROLL_MAX, bool USEFASTMATH, typename real_t>
__global__ void real_euler_kernel(real_t *dst, const uint32_t ops) {
	real_t acc = 0;
	real_t input_i = common_float_input[threadIdx.x];

#pragma unroll UNROLL_MAX
	for (uint32_t count = 0; count < ops; count++) {
		acc += euler<USEFASTMATH>(-input_i);
	}

	dst[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

template<typename real_t>
void execute_kernel(MICROINSTRUCTION& micro, real_t* output, size_t grid_size,
		size_t block_size, uint32_t operation_num, bool fast_math) {

	//	real_t, real_t,
	void (*kernel)(real_t*, uint32_t);
	switch (micro) {
	case ADD:
		kernel = real_add_kernel<LOOPING_UNROLL, true>;
		break;
	case MUL:
		kernel = real_mul_kernel<LOOPING_UNROLL, true>;
		break;
	case FMA:
		kernel = real_fma_kernel<LOOPING_UNROLL, true>;
		break;
	case DIV:
		if (fast_math) {
			kernel = real_div_kernel<LOOPING_UNROLL, true>;
		} else {
			kernel = real_div_kernel<LOOPING_UNROLL, false>;
		}

		break;
	case PYTHAGOREAN:
		if (fast_math) {
			kernel = real_pythagorean_kernel<LOOPING_UNROLL, true>;

		} else {
			kernel = real_pythagorean_kernel<LOOPING_UNROLL, false>;
		}
		break;
	case EULER:
		if (fast_math) {
			kernel = real_euler_kernel<LOOPING_UNROLL, true>;

		} else {
			kernel = real_euler_kernel<LOOPING_UNROLL, false>;

		}
		break;
	}

	kernel<<<grid_size, block_size>>>(output, operation_num);
}

template<>
void MicroReal<float>::execute_micro() {
	execute_kernel(this->parameters.micro, this->output_device_1.data(),
			this->parameters.grid_size, this->parameters.block_size,
			this->parameters.operation_num, this->parameters.fast_math);

}
