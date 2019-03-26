__global__ void MicroBenchmarkKernel_FMA(tested_type *d_R0) {
// ========================================== Double and Single precision
#if defined(test_precision_double) or defined(test_precision_single)
	register tested_type acc = OUTPUT_R;
	register tested_type input_a = INPUT_A;
	register tested_type input_b = INPUT_B;
	register tested_type input_a_neg = -INPUT_A;
	register tested_type input_b_neg = -INPUT_B;
#elif defined(test_precision_half)
	register half2 acc = __float2half2_rn(OUTPUT_R);
	register half2 input_a = __float2half2_rn(INPUT_A);
	register half2 input_b = __float2half2_rn(INPUT_B);
	register half2 input_a_neg = __float2half2_rn(-INPUT_A);
	register half2 input_b_neg = __float2half2_rn(-INPUT_B);
	register half2 *d_R0_half2 = (half2*)d_R0;
	//register half2 *d_R1_half2 = (half2*)d_R1;
	//register half2 *d_R2_half2 = (half2*)d_R2;
#endif

#pragma unroll 512
	for (register unsigned int count = 0;
			count < (OPS / (4 * OPS_PER_THREAD_OPERATION)); count++) {
#if defined(test_precision_double)
		acc = fma(input_a, input_b, acc);
		acc = fma(input_a_neg, input_b, acc);
		acc = fma(input_a, input_b_neg, acc);
		acc = fma(input_a_neg, input_b_neg, acc);
#elif defined(test_precision_single)
		acc = __fmaf_rn(input_a, input_b, acc);
		acc = __fmaf_rn(input_a_neg, input_b, acc);
		acc = __fmaf_rn(input_a, input_b_neg, acc);
		acc = __fmaf_rn(input_a_neg, input_b_neg, acc);
#elif defined(test_precision_half)
		acc = __hfma2(input_a, input_b, acc);
		acc = __hfma2(input_a_neg, input_b, acc);
		acc = __hfma2(input_a, input_b_neg, acc);
		acc = __hfma2(input_a_neg, input_b_neg, acc);
#endif
	}

#if defined(test_precision_double) or defined(test_precision_single)
	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif defined(test_precision_half)
	d_R0_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}



__global__ void MicroBenchmarkKernel_ADD(tested_type *d_R0) {
// ========================================== Double and Single precision
#if defined(test_precision_double) or defined(test_precision_single)
	register tested_type acc = OUTPUT_R;
	register tested_type input_a = OUTPUT_R;
	register tested_type input_a_neg = -OUTPUT_R;
#elif defined(test_precision_half)
	register half2 acc = __float2half2_rn(OUTPUT_R);
	register half2 input_a = __float2half2_rn(OUTPUT_R);
	register half2 input_a_neg = __float2half2_rn(-OUTPUT_R);
	register half2 *d_R0_half2 = (half2*)d_R0;
	//register half2 *d_R1_half2 = (half2*)d_R1;
	//register half2 *d_R2_half2 = (half2*)d_R2;
#endif

#pragma unroll 512
	for (register unsigned int count = 0;
			count < (OPS / (4 * OPS_PER_THREAD_OPERATION)); count++) {
#if defined(test_precision_double)
		acc = __dadd_rn(acc, input_a);
		acc = __dadd_rn(acc, input_a_neg);
		acc = __dadd_rn(acc, input_a_neg);
		acc = __dadd_rn(acc, input_a);
#elif defined(test_precision_single)
		acc = __fadd_rn(acc, input_a);
		acc = __fadd_rn(acc, input_a_neg);
		acc = __fadd_rn(acc, input_a_neg);
		acc = __fadd_rn(acc, input_a);
#elif defined(test_precision_half)
		acc = __hadd2(acc, input_a);
		acc = __hadd2(acc, input_a_neg);
		acc = __hadd2(acc, input_a_neg);
		acc = __hadd2(acc, input_a);
#endif
	}

#if defined(test_precision_double) or defined(test_precision_single)
	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif defined(test_precision_half)
	d_R0_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}

__global__ void MicroBenchmarkKernel_MUL(tested_type *d_R0) {
// ========================================== Double and Single precision
#if defined(test_precision_double) or defined(test_precision_single)
	register tested_type acc = OUTPUT_R;
	register tested_type input_a = INPUT_A;
	register tested_type input_a_inv = 1.0/INPUT_A;
#elif defined(test_precision_half)
	register half2 acc = __float2half2_rn(OUTPUT_R);
	register half2 input_a = __float2half2_rn(INPUT_B);
	register half2 input_a_inv = __float2half2_rn(1.0/INPUT_B);
	register half2 *d_R0_half2 = (half2*)d_R0;
	//register half2 *d_R1_half2 = (half2*)d_R1;
	//register half2 *d_R2_half2 = (half2*)d_R2;
#endif

#pragma unroll 512
	for (register unsigned int count = 0;
			count < (OPS / (4 * OPS_PER_THREAD_OPERATION)); count++) {
#if defined(test_precision_double)
		acc = __dmul_rn(acc, input_a);
		acc = __dmul_rn(acc, input_a_inv);
		acc = __dmul_rn(acc, input_a_inv);
		acc = __dmul_rn(acc, input_a);
#elif defined(test_precision_single)
		acc = __fmul_rn(acc, input_a);
		acc = __fmul_rn(acc, input_a_inv);
		acc = __fmul_rn(acc, input_a_inv);
		acc = __fmul_rn(acc, input_a);
#elif defined(test_precision_half)
		acc = __hmul2(acc, input_a);
		acc = __hmul2(acc, input_a_inv);
		acc = __hmul2(acc, input_a_inv);
		acc = __hmul2(acc, input_a);
#endif
	}

#if defined(test_precision_double) or defined(test_precision_single)
	d_R0[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#elif defined(test_precision_half)
	d_R0_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R1_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
//	d_R2_half2[blockIdx.x * blockDim.x + threadIdx.x] = acc;
#endif
}







