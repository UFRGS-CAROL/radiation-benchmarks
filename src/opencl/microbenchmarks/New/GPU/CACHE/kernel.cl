
#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#include "/home/carol/radiation-benchmarks/src/opencl/microbenchmarks/New/GPU/CACHE/support/common.h"

#pragma OPENCL EXTENSION cl_amd_fp64: enable
__kernel void
cache(__global T* A,
          __global T* B,
          __global T* C,
          int n)
{
	int fator = 1 * get_global_id(0);

    int tx = fator +  get_global_id(1);

    double value = A[tx];
    for (int k = 0; k < n; ++k) {
        value += B[tx];
    }
    C[tx] = value;
}

