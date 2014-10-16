
// OpenCL Kernel
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void
simpleDiv(__global double* A,
__global double* B,
int n)
{

	int tx = get_global_id(0);

	double value = A[tx];
	for (int k = 0; k < n; ++k) {
		value /= B[tx];
	}

	A[tx] = value;

}

#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void
checkGold(__global double* A,
__global double* GOLD,
__global int *ea
)
{

	int tx = get_global_id(0);

	private int gold = GOLD[tx];
	private int value = A[tx];
	if ((fabs((float)(value- gold )/value) > 0.0000000001)||(fabs((float)(value-gold)/gold) > 0.0000000001)){
	//if(A[tx] != GOLD[tx]){
		atomic_inc(ea);
	}
}

