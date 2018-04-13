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

#include <unistd.h>
#include <thread>
#include <assert.h>

//*****************************************  LOG  ***********************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************************//
/*
inline int new_compare_output(XYZ *outp, XYZ *outpCPU, int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
	int errors=0;

    double sum_delta2_x,sum_delta2_y,sum_delta2_z, sum_ref2, L1norm2;
    sum_delta2_x = 0;
    sum_delta2_y = 0;
    sum_delta2_z = 0;

    sum_ref2   = 0;
    L1norm2    = 0;
    for(int i = 0; i < RESOLUTIONI; i++) {
        for(int j = 0; j < RESOLUTIONJ; j++) {

			sum_delta2_x = fabs(outp[i * RESOLUTIONJ + j].x - outpCPU[i * RESOLUTIONJ + j].x) / fabs(outpCPU[i * RESOLUTIONJ + j].x);
			sum_delta2_y = fabs(outp[i * RESOLUTIONJ + j].y - outpCPU[i * RESOLUTIONJ + j].y) / fabs(outpCPU[i * RESOLUTIONJ + j].y);
			sum_delta2_z = fabs(outp[i * RESOLUTIONJ + j].z - outpCPU[i * RESOLUTIONJ + j].z) / fabs(outpCPU[i * RESOLUTIONJ + j].z);

			if(sum_delta2_x >= 1e-8 || sum_delta2_y >= 1e-8 || sum_delta2_z >= 1e-8  ){

		        errors++;
#ifdef LOGS
		        char error_detail[200];
        		sprintf(error_detail," p: [%d, %d]; X r: %f, e: %f ; Y: r: %f, e: %f ; Z: r: %f, e: %f  ",i,j,outp[i * RESOLUTIONJ + j].x,outpCPU[i * RESOLUTIONJ + j].x,outp[i * RESOLUTIONJ + j].y,outpCPU[i * RESOLUTIONJ + j].y,outp[i * RESOLUTIONJ + j].z,outpCPU[i * RESOLUTIONJ + j].z);

       			 log_error_detail(error_detail);
#endif			

			}

        }
    }

    return errors;
}
*/
// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_work_groups;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    int         in_size_i;
    int         in_size_j;
    int         out_size_i;
    int         out_size_j;

    Params(int argc, char **argv) {
/*
//CPU
        platform      = 0;
        device        = 0;
        n_work_items  = 5;
        n_work_groups = 5;
        n_threads     = 4;
        n_warmup      = 5;
        n_reps        = 1;
        alpha         = 1.0;
        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 2500;
*/

/*
// GPU
        platform      = 0;
        device        = 0;
        n_work_items  = 16;
        n_work_groups = 8;
        n_threads     = 1;
        n_warmup      = 5;
        n_reps        = 1;
        alpha         = 0.0;
        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 2500;

*/

// CPU+GPU 
        platform      = 0;
        device        = 0;
        n_work_items  = 16;
        n_work_groups = 8;
        n_threads     = 2;
        n_warmup      = 5;
        n_reps        = 1;
        alpha         = 0.1;
        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 2500;

        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform      = atoi(optarg); break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_work_items  = atoi(optarg); break;
            case 'g': n_work_groups = atoi(optarg); break;
            case 't': n_threads     = atoi(optarg); break;
            case 'w': n_warmup      = atoi(optarg); break;
            case 'r': n_reps        = atoi(optarg); break;
            case 'a': alpha         = atof(optarg); break;
            case 'f': file_name     = optarg; break;
            case 'm': in_size_i = in_size_j = atoi(optarg); break;
            case 'n': out_size_i = out_size_j = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
            assert(n_work_groups > 0 && "Invalid # of device work-groups!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
            assert(n_work_groups > 0 && "Invalid # of device work-groups!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef OCL_2_0
            assert((n_work_items > 0 && n_work_groups > 0 || n_threads > 0) && "Invalid # of host + device workers!");
#else
            assert(0 && "Illegal value for -a");
#endif
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./bs [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=16)"
                "\n    -g <G>    # of device work-groups (default=32)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of output elements to process on host (default=0.1)"
#ifdef OCL_2_0
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
#else
                "\n              NOTE: <A> must be between 0.0 and 1.0"
#endif
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    name of input file with control points (default=input/control.txt)"
                "\n    -m <N>    input size in both dimensions (default=3)"
                "\n    -n <R>    output resolution in both dimensions (default=300)"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void new_read_input(XYZ *in, const Params &p) {

    // Open input file
    FILE *f = NULL;
    char filename[300];
    snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/input_%d",p.out_size_i); // Gold com a resolução 
    int in_size   = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
    FILE *finput;
    if (finput = fopen(filename, "rb")) {
        fread(in, in_size, 1 , finput);
    } else {
        printf("Error reading input file\n");
        exit(1);
    }
	fclose(finput);	

}

// Main -----------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    OpenCLSetup  ocl(p.platform, p.device);
    Timer        timer;
    cl_int       clStatus;
	int err = 0;

printf("-p %d -d %d -i %d -g %d -a %.2f -t %d \n",p.platform , p.device, p.n_work_items, p.n_work_groups,p.alpha,p.n_threads);


#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300, "-i %d -g %d -a %.2f -t %d -n %d",p.n_work_items, p.n_work_groups,p.alpha,p.n_threads, p.out_size_i);
    start_log_file("openclBezierSurface", test_info);
	//printf("Com LOG\n");
#endif


    // Allocate
    timer.start("Allocation");
    int in_size   = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
    int out_size  = p.out_size_i * p.out_size_j * sizeof(XYZ);
    int n_tasks_i = divceil(p.out_size_i, p.n_work_items);
    int n_tasks_j = divceil(p.out_size_j, p.n_work_items);
    int n_tasks   = n_tasks_i * n_tasks_j;
#ifdef OCL_2_0
    XYZ *h_in     = (XYZ *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, in_size, 0);
    XYZ *h_out    = (XYZ *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, out_size, 0);
    XYZ *d_in     = h_in;
    XYZ *d_out    = h_out;
    std::atomic_int *worklist = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    ALLOC_ERR(h_in, h_out, worklist);
#else
    XYZ *  h_in        = (XYZ *)malloc(in_size);
    XYZ *  h_out       = (XYZ *)malloc(out_size);
    XYZ *  h_out_merge = (XYZ *)malloc(out_size);
    XYZ *  gold = (XYZ *)malloc(out_size);

    cl_mem d_in  = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, in_size, NULL, &clStatus);
    cl_mem d_out = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, out_size, NULL, &clStatus);
    CL_ERR();
    ALLOC_ERR(h_in, h_out, h_out_merge);
#endif
    clFinish(ocl.clCommandQueue);
    timer.stop("Allocation");
    //timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
//	printf("Maximum Work Items %d\n",max_wi);
    new_read_input(h_in, p);

// *********************** Lendo GOLD   *****************************
    char filename[300];
    snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/gold_%d",p.out_size_i); // Gold com a resolução 
    FILE *finput;
    if (finput = fopen(filename, "rb")) {
        fread(gold, out_size, 1 , finput);
    } else {
        printf("Error reading gold file\n");
        exit(1);
    }
	fclose(finput);	


    clFinish(ocl.clCommandQueue);
    timer.stop("Initialization");
    //timer.print("Initialization", 1);
    // Loop over main kernel -- Vamos executar somente uma vez o kernel
    for(int rep = 0; rep < p.n_reps; rep++) {

//printf("Rep\n");
#ifndef OCL_2_0
    // Copy to device
    timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(ocl.clCommandQueue, d_in, CL_TRUE, 0, in_size, h_in, 0, NULL, NULL);
    clFinish(ocl.clCommandQueue);
    CL_ERR();
    timer.stop("Copy To Device");
    //timer.print("Copy To Device", 1);
#endif


// Reset
#ifdef OCL_2_0
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#endif

        //if(rep >= p.n_warmup)
        timer.start("Kernel");

        // Launch GPU threads
        clSetKernelArg(ocl.clKernel, 0, sizeof(int), &n_tasks);
        clSetKernelArg(ocl.clKernel, 1, sizeof(float), &p.alpha);
        clSetKernelArg(ocl.clKernel, 2, sizeof(int), &p.in_size_i);
        clSetKernelArg(ocl.clKernel, 3, sizeof(int), &p.in_size_j);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &p.out_size_i);
        clSetKernelArg(ocl.clKernel, 5, sizeof(int), &p.out_size_j);
        clSetKernelArg(ocl.clKernel, 6, in_size, NULL);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 7, d_in);
        clSetKernelArgSVMPointer(ocl.clKernel, 8, d_out);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, worklist);
        clSetKernelArg(ocl.clKernel, 10, sizeof(int), NULL);
#else
        clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_in);
        clSetKernelArg(ocl.clKernel, 8, sizeof(cl_mem), &d_out);
#endif

        // Kernel launch
        size_t ls[2] = {(size_t)p.n_work_items, (size_t)p.n_work_items};
        size_t gs[2] = {(size_t)p.n_work_items * p.n_work_groups, (size_t)p.n_work_items};
        if(p.n_work_groups > 0) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
#ifdef LOGS
        start_iteration();
#endif
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 2, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in, h_out, n_tasks, p.alpha, p.n_threads, p.n_work_items, p.in_size_i,
            p.in_size_j, p.out_size_i, p.out_size_j
#ifdef OCL_2_0
            ,
            worklist
#endif
            );

        clFinish(ocl.clCommandQueue);
        main_thread.join();

        //if(rep >= p.n_warmup)
       timer.stop("Kernel");
 
  //}
    //timer.print("Kernel",1);


#ifdef LOGS
        end_iteration();
#endif

#ifndef OCL_2_0
    // Copy back
    timer.start("Copy Back and Merge");
    clStatus = clEnqueueReadBuffer(ocl.clCommandQueue, d_out, CL_TRUE, 0, out_size, h_out_merge, 0, NULL, NULL);
    CL_ERR();
    clFinish(ocl.clCommandQueue);
    // Merge
    int cut = n_tasks * p.alpha;
    for(int t = 0; t < cut; ++t) {
        const int ty  = t / n_tasks_j;
        const int tx  = t % n_tasks_j;
        int       row = ty * p.n_work_items;
        int       col = tx * p.n_work_items;
        for(int i = row; i < row + p.n_work_items; ++i) {
            for(int j = col; j < col + p.n_work_items; ++j) {
                if(i < p.out_size_i && j < p.out_size_j) {
                    h_out_merge[i * p.out_size_j + j] = h_out[i * p.out_size_j + j];
                }
            }
        }
    }
    timer.stop("Copy Back and Merge");
    //timer.print("Copy Back and Merge", 1);
#endif

// Verify answer
#ifdef OCL_2_0
	printf("OpenCL 2.0 not supported by compare_output - Developing");
	exit(0);
#else
   err= new_compare_output(h_out_merge, gold, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);
#endif

        if(err > 0) {
            printf("Errors: %d\n",err);
			snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/BS/gold_%d",p.out_size_i); // Gold com a resolução 
			if (finput = fopen(filename, "rb")) {
				fread(gold, out_size, 1 , finput);
			} else {
				printf("Error reading gold file\n");
				exit(1);
			}
			fclose(finput);	
		new_read_input(h_in,p);
        } else {
            printf(".");
        }
//    	new_read_input(h_in, p);

#ifdef LOGS
        log_error_count(err);
#endif

	}

#ifdef LOGS
    end_log_file();
#endif

    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in);
    clSVMFree(ocl.clContext, h_out);
    clSVMFree(ocl.clContext, worklist);
#else
    free(h_in);
    free(h_out);
    free(h_out_merge);
    clStatus = clReleaseMemObject(d_in);
    clStatus = clReleaseMemObject(d_out);
    CL_ERR();
#endif
    ocl.release();
    timer.stop("Deallocation");
    //timer.print("Deallocation", 1);

   // printf("Test Passed\n");
    return 0;
}
