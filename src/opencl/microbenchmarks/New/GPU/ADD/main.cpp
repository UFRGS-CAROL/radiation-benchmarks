/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include "kernel.h"
#include "support/common.h"
#include "support/ocl.h"
#include "support/timer.h"
#include "support/verify.h"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>

//*****************************************  LOG  ***********************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************************//

// Params ---------------------------------------------------------------------
struct Params {

    int   platform;
    int   device;
    int   n_work_items;
    int   n_work_groups;
    int   n_reps;

    Params(int argc, char **argv) {
        platform          = 0;
        device            = 0;
        n_work_items      = 256;
        n_work_groups     = 8;
        n_reps            = 1;

        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:n:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform          = atoi(optarg); break;
            case 'd': device            = atoi(optarg); break;
            case 'i': n_work_items      = atoi(optarg); break;
            case 'g': n_work_groups     = atoi(optarg); break;
            case 'r': n_reps            = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./sc [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=256)"
                "\n    -g <G>    # of device work-groups (default=8)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.1)"
#ifdef OCL_2_0
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -n <N>    input size (default=1048576)"
                "\n    -c <C>    compaction factor (default=50)"
                "\n");
    }
};

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;
	int err = 0;

    printf("-p %d -d %d -i %d -g %d -a %.2f -t %d -n %d -c %d \n",p.platform , p.device, p.n_work_items,p.n_work_groups,p.alpha,p.n_threads,p.in_size,p.compaction_factor);
	
	//printf("Main size:%d\n",p.in_size);
	
#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300, "-i %d -g %d -a %.2f -t %d -n %d -c %d",p.n_work_items,        p.n_work_groups,p.alpha,p.n_threads,p.in_size,p.compaction_factor);
    start_log_file("MicroBenchmark_ADD_INT_GPU", test_info);
#endif

    int error = 0;
    int i = 0;
    int n = 10;                                 // Times a workitem will sum
    int size = p.n_work_items*p.n_work_items;    // testar tamanho maximo
    int gold = 5 + n*5;
    T *A = ( T* ) malloc( size * sizeof( T ) );
    T *B = ( T* ) malloc( size * sizeof( T ) );
    T *C = ( T* ) malloc( size * sizeof( T ) );

    // Initialize memory 
    for(i=0;i<size;i++){
        A[i] = 5;
        B[i] = 5;
        C[i] = 0;         
     }
    
    cl_mem d_A = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    cl_mem d_B = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    cl_mem d_C = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    CL_ERR();    
    clFinish(ocl.clCommandQueue);

    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_A, CL_TRUE, 0, size * sizeof(T),A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_B, CL_TRUE, 0, size * sizeof(T),B, 0, NULL, NULL);        
    CL_ERR();
    clFinish(ocl.clCommandQueue);
 
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    printf("Max WorkItems:%d\n",max_wi);    
    
    for(int rep = 0; rep < p.n_reps; rep++) {

        clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_C, CL_TRUE, 0, size * sizeof(T),C, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);
            
        clSetKernelArg(ocl.clKernel, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(ocl.clKernel, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(ocl.clKernel, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &n);

        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_items * p.n_work_groups};
        if(gs[0] > 0) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");      
        }
        clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);

        // Verificar se é encessario, pq alocamos host reachable
        clStatus   = clEnqueueReadBuffer(ocl.clCommandQueue, d_C, CL_TRUE, 0,  size * sizeof(T), C, 0, NULL, NULL);
        clFinish(ocl.clCommandQueue);
        
        
        // Verify errors
        error = 0;
        for(i=0;i<size;i++){
           if( C[i] != gold){
                // logar o erro
                error ++;
           }
           C[i] = 0 ;        
        }
        if(error != 0){
            printf("Deu ruim %d\n",erorr);
        }
        else{
            printf(".");
        }        

    }
  
    // Free memory
    free(A);
    free(B);
    free(C);
    clStatus = clReleaseMemObject(d_A);
    clStatus = clReleaseMemObject(d_B);
    clStatus = clReleaseMemObject(d_C);    
    CL_ERR();
    ocl.release();
    printf("Test Passed\n");
    return 0;
}
