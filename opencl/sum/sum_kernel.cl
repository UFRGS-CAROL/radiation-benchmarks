
// OpenCL Kernel
//#pragma OPENCL EXTENSION cl_khr_fp64: enable
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


	double gold = A[tx] + n*B[tx];
	if ((fabs((value- gold )/value) > 0.0000000001)||(fabs((value-gold)/gold) > 0.0000000001))
	//if ((A[tx] + n*B[tx]) != value)
		atomic_inc(kernel_errors);

	A[tx] = value;

}

/*
__kernel void
checkGold(__global double* A,
__global double* GOLD,
__global int *ea
)
{

	int tx = get_global_id(0);

	if(A[tx] != GOLD[tx]){
		atomic_inc(ea);
	}
}
*/
