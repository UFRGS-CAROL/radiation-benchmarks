#define ROTATE_ITERATION 10000

// OpenCL Kernel
__kernel void simpleRotate(int refword, int nreps, __global int *kernel_errors)
{

    int testVar = refword;

    for(int i=0; i<nreps; i++)
    {
        for (int j=0; j<ROTATE_ITERATION; j++)
        {
            rotate(testVar, 1);

// injecting one error
//int tx = get_global_id(0);
//if (tx == 0 && i==0 && j==0)
//	testVar=refword+2;

            // 32 rotates and testVar must be equal to refword
            if ( j % 32 == 0 && testVar != refword) {
                atomic_inc(kernel_errors);

                testVar = refword;
            }
        }
    }

}


