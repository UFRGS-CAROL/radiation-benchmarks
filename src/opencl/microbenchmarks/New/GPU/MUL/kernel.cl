
#define _OPENCL_COMPILER_

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

#include "/home/carol/radiation-benchmarks/src/opencl/microbenchmarks/New/GPU/MUL/support/common.h"

#pragma OPENCL EXTENSION cl_amd_fp64: enable
__kernel void
simpleSumINT(__global T* A,
          __global T* B,
          __global T* C,
          int n)
{
	int fator = 1 * get_global_id(0);

    int tx = fator +  get_global_id(1);

    double value = A[tx];
    for (int k = 0; k < n/10; ++k) {

        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value /= 9765625;

    }
    C[tx] = value;
}

__kernel void
simpleSumFLOAT(__global T* A,
          __global T* B,
          __global T* C,
          int n)
{
	int fator = 1 * get_global_id(0);

    int tx = fator +  get_global_id(1);

    double value = A[tx];
    for (int k = 0; k < n/10; ++k) {
	value *= B[tx];
	value *=6.8212105343475049e-14;
        value *= B[tx];
        value *=6.8212105343475049e-14;
        value *= B[tx];
        value *=6.8212105343475049e-14;
        value *= B[tx];
        value *=6.8212105343475049e-14;
        value *= B[tx];
        value *=6.8212105343475049e-14;

/*
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value *= B[tx];
        value /= 25329516.211914063;
*/  
  }
    C[tx] = value;
}



