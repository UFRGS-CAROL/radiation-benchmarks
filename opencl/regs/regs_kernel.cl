
//define how many register each thread will use
#define TEST_REGS_PER_THREAD 247

// OpenCL Kernel
__kernel void
testRegs(int refword,
int nreps,
__global int *kernel_errors
)
{

	int tx = get_global_id(0);
	
	__private int local_test_vector[TEST_REGS_PER_THREAD];


	// initialize regs
	#pragma unroll
	for (int i=0; i<TEST_REGS_PER_THREAD; i++)
	{
		local_test_vector[i] = refword;
	}


	// code that should never run.
	// Prevents the compiler from completely optimizing away local_test_vector
	if (tx == 1323513)
	{
		#pragma unroll
		for (int i=0; i<TEST_REGS_PER_THREAD; i++)
			local_test_vector[i] += local_test_vector[tx+i] & 0xffffffff;
	}

	// test loop
	for(int i=0; i<nreps; i++)
	{

		#pragma unroll
		for (int j=0; j<TEST_REGS_PER_THREAD; j++)
		{
			if (local_test_vector[j] != refword) {
				atomic_inc(kernel_errors);
				local_test_vector[j] = refword;
			}
		}
	}


}


