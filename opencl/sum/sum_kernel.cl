
// OpenCL Kernel
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_amd_fp64: enable
__kernel void
simpleSum(__global double* A,
          __global double* B,
          __global int *kernel_errors,
          int n)
{

    int tx = get_global_id(0);

    double value = A[tx];
    for (int k = 0; k < n; ++k) {
        value += B[tx];
    }

// injecting one error
//if (tx == 0)
//	value=1;

    // Check if value computed is different of GOLD[tx] within some precision
    double gold = A[tx] + n*B[tx];
    if ((fabs((value- gold )/value) > 0.0000000001)||(fabs((value-gold)/gold) > 0.0000000001))
        atomic_inc(kernel_errors);

    A[tx] = value;

}

