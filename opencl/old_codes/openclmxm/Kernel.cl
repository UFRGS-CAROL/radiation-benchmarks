// kernel.cl
// Multiply two matrices A * B = C
// Device code.

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#pragma OPENCL EXTENSION cl_khr_fp64: enable
// Note: no support on Intel HD4600 integrated graphics
 
// OpenCL Kernel
__kernel void
matrixMul(__global double* A, __global double* B, __global double* C, int size)
{
   int tx = get_global_id(0);
   int ty = get_global_id(1);
 
   __private double value = 0;
   for (int k = 0; k < size; ++k)
   {
      value += A[ty * size + k] * B[k * size + tx];
   }
 
   C[ty * size + tx] = value;
}

__kernel void
GoldChk(__global double* GOLD, __global double* C, int size, __global int* kerrors)
{
   int tx = get_global_id(0);
   int ty = get_global_id(1);
 
   if ((fabs((GOLD[ty*size+tx]-C[ty*size+tx])/GOLD[ty*size+tx]) > 0.0000000001)||(fabs((GOLD[ty*size+tx]-C[ty*size+tx])/C[ty*size+tx]) > 0.0000000001))
	   atomic_inc(kerrors);
}
