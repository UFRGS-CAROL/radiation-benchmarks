
//define how many register each thread will use
#define TEST_ARRAY_SIZE 15*1024*10

// OpenCL Kernel
__kernel void
testCache(__global int* g_test_array,
__global int* g_output_array,
int stride,
int refword,
int nreps,
__global int *kernel_errors
)
{

	int tx = get_global_id(0);

	// initialize test array
	for (int i=tx; i<TEST_ARRAY_SIZE; i+=stride)
	{
		g_test_array[i] = refword;
	}



	// test loop
	for(int i=0; i<nreps; i++)
	{

		for (int j=tx; j<TEST_ARRAY_SIZE; j+=stride)
		{
			int temp = g_test_array[j];
			if (temp != refword) {
				// store old values of kernel_errors in idx
				__private int idx = atomic_inc(kernel_errors);

				//saves iteration, position and xor of error found
				idx = (idx) * 3;
				g_output_array[idx] = i;
				g_output_array[idx+1] = j;
				g_output_array[idx+2] = temp ^ refword;

				g_test_array[j] = refword;
			}
		}
	}


}


