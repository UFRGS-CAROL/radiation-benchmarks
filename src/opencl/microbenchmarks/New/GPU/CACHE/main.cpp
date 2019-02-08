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

//#include "kernel.h"
#include "support/common.h"
#include "support/ocl.h"
#include "support/timer.h"
#include <string.h>
#include <unistd.h>
#include <thread>
#include <assert.h>
#include <math.h>

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
	int   op; 	

    Params(int argc, char **argv) {
        platform          = 0;
        device            = 0;
        n_work_items      = 256;
        n_work_groups     = 8;
        n_reps            = 1;
		op 				  = 500;		
        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:n:c:o:")) >= 0) {
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
            case 'o': op	            = atoi(optarg); break;
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
                "\n    -o <O>    # of operations performed per WorkItem(GPU Thread)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
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

    printf("-p %d -d %d -i %d -g %d -o %d\n",p.platform , p.device, p.n_work_items,p.n_work_groups,p.op);
	
	//printf("Main size:%d\n",p.in_size);
	
#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300, "-i %d -g %d -o %d\n ",p.n_work_items,p.n_work_groups,p.op);
    start_log_file("MicroBenchmark_CACHE_GPU", test_info);
#endif

    int error = 0;
    int i = 0;
    int n = p.op;                                 												// Operations per WorkItem
    int size = p.n_work_items*p.n_work_items *p.n_work_items*p.n_work_groups ;    				// testar tamanho maximo
    int gold = 500;//5 + n*5;
	unsigned char* temp ;
	// Allocando um Vetor de ponteiros, cada elemento aponta para um vetor de 32K Bytes
	unsigned char **Buffer = (unsigned char**) malloc(size* sizeof(unsigned char*));
	for(i=0;i<size;i++){
		temp  = (unsigned char*) malloc(32768*sizeof(unsigned char));
		memset(temp,0xFF,32768*sizeof(unsigned char));
		Buffer[i] = temp;
	}


    cl_mem d_mem = clCreateBuffer(ocl.clContext,  CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, size*sizeof(unsigned char*), NULL, &clStatus);
    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_mem, CL_TRUE, 0, size*sizeof(unsigned char*) ,A, 0, NULL, NULL);

    cl_mem d_interm_gpu_proxy = clCreateBuffer(ocl.clContext,  CL_MEM_READ_WRITE| CL_MEM_USE_HOST_PTR , in_size, d_interm_gpu_proxy_2, &clStatus);

    // Initialize memory

    for(i=0;i<size;i++){
        A[i] = 0;
        B[i] = 1;
        C[i] = 0;         
     }


    
    cl_mem d_A = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    cl_mem d_B = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    cl_mem d_C = clCreateBuffer( ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size * sizeof(T), NULL, &clStatus);
    CL_ERR();    
    clFinish(ocl.clCommandQueue);

    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_A, CL_TRUE, 0, size * sizeof(T),A, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_B, CL_TRUE, 0, size * sizeof(T),B, 0, NULL, NULL);        
    clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_C, CL_TRUE, 0, size * sizeof(T),C, 0, NULL, NULL);            
	CL_ERR();
    clFinish(ocl.clCommandQueue);
 
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    printf("Max WorkItems:%d\n",max_wi); 
	printf("Gold:%d\n",gold);  

    
    for(int rep = 0; rep < p.n_reps; rep++) {



        clStatus = clEnqueueWriteBuffer( ocl.clCommandQueue, d_C, CL_TRUE, 0, size * sizeof(T),C, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);
            
        clSetKernelArg(ocl.clKernel, 0, sizeof(cl_mem), &d_A);
        clSetKernelArg(ocl.clKernel, 1, sizeof(cl_mem), &d_B);
        clSetKernelArg(ocl.clKernel, 2, sizeof(cl_mem), &d_C);
        clSetKernelArg(ocl.clKernel, 3, sizeof(int), &n);

        size_t ls[2] = {(size_t)p.n_work_items,(size_t)p.n_work_items};
        size_t gs[2] = {(size_t)p.n_work_items*p.n_work_items *p.n_work_items*p.n_work_groups,(size_t)p.n_work_items*p.n_work_items*p.n_work_items*p.n_work_groups};
        if(gs[0] > 0) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");      
        }
#ifdef LOGS
	start_iteration();
#endif
        clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);
#ifdef LOGS
	end_iteration();
#endif

        clStatus   = clEnqueueReadBuffer(ocl.clCommandQueue, d_C, CL_TRUE, 0,  size * sizeof(T), C, 0, NULL, NULL);
        clFinish(ocl.clCommandQueue);
        
        
        // Verify errors    
		error = 0;
#pragma omp parallel for private(i)	
        for(i=0;i<size;i++){


            if( C[i] != gold){
                error ++;
#ifdef LOGS
				char error_detail[200];
				sprintf(error_detail,"i=%d,E=%d ,R=%d",i,gold,C[i]);
				log_error_detail(error_detail);
#endif		
           }
           C[i] = 0 ;      // Cleaning output for next iteration  
        }
        if(error != 0){
            printf("Deu ruim %d\n",error);
        }
        else{
            printf(".");
			fflush(stdout);
        }  
#ifdef LOGS
	    log_error_count(error); 	// Always just one error.
#endif      

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