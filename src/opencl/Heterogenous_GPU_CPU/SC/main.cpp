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
    int   n_threads;
    int   n_warmup;
    int   n_reps;
    float alpha;
    int   in_size;
    int   compaction_factor;
    int   remove_value;

    Params(int argc, char **argv) {
/*
//CPU
        platform          = 0;
        device            = 0;
        n_work_items      = 256;
        n_work_groups     = 8;
        n_threads         = 4;
        n_warmup          = 0;
        n_reps            = 1;
        alpha             = 1.0;
        in_size           = 367001600;
        compaction_factor = 50;
        remove_value      = 0;

*/
/*
//GPU
        platform          = 0;
        device            = 0;
        n_work_items      = 256;
        n_work_groups     = 8;
        n_threads         = 4;
        n_warmup          = 0;
        n_reps            = 1;
        alpha             = 0.0;
        in_size           = 367001600;
        compaction_factor = 50;
        remove_value      = 0;
*/

//CPU + GPU
        platform          = 0;
        device            = 0;
        n_work_items      = 256;
        n_work_groups     = 8;
        n_threads         = 4;
        n_warmup          = 0;
        n_reps            = 1;
        alpha             = 0.7;
        in_size           = 367001600;
        compaction_factor = 50;
        remove_value      = 0;

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
            case 't': n_threads         = atoi(optarg); break;
            case 'w': n_warmup          = atoi(optarg); break;
            case 'r': n_reps            = atoi(optarg); break;
            case 'a': alpha             = atof(optarg); break;
            case 'n': in_size           = atoi(optarg); break;
            case 'c': compaction_factor = atoi(optarg); break;
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

inline int create_output(T *ouput, const Params &p){
	const int n_tasks     = divceil(p.in_size, p.n_work_items * REGS);
	int in_size = n_tasks * p.n_work_items * REGS * sizeof(T);
	char filename[100];
	FILE *finput;
	snprintf(filename, 100, "gold_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
	if (finput = fopen(filename, "wb")) {
    	fwrite(ouput, in_size, 1 , finput);
	} else {
    	printf("Error writing output file");
    	exit(1);
	}
	fclose(finput);

}


// Input Data -----------------------------------------------------------------
void new_read_input(T *input, const Params &p) {

    FILE *f = NULL;
    char filename[300];
    snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/SC/input_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
    const int n_tasks     = divceil(p.in_size, p.n_work_items * REGS);
    int in_size   = n_tasks * p.n_work_items * REGS * sizeof(T);
    FILE *finput;
    if (finput = fopen(filename, "rb")) {
        fread(input, in_size, 1 , finput);
    } else {
        printf("Error reading input file\n");
        exit(1);
    }
	fclose(finput);	
}

void read_input(T *input, const Params &p) {

    // Initialize the host input vectors
    srand(15);			// Colocamos  a mesma semente para gerar a mesma Sequencia de Números 
    for(int i = 0; i < p.in_size; i++) {
        input[i] = (T)p.remove_value;
    }
	update_timestamp();
    int M = (p.in_size * p.compaction_factor) / 100;
    int m = M;
    while(m > 0) {
        int x = (int)(p.in_size * (((float)rand() / (float)RAND_MAX)));
        if(x < p.in_size)
            if(input[x] == p.remove_value) {
                input[x] = (T)(x + 2);
                m--;
            }
    }
}

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
    snprintf(test_info, 300, "-i %d -g %d -a %.2f -t %d -n %d -c %d",p.n_work_items, p.n_work_groups,p.alpha,p.n_threads,p.in_size,p.compaction_factor);
    start_log_file("openclStreamCompaction", test_info);
	//printf("Com LOG\n");
#endif


    // Allocate buffers
    //timer.start("Allocation");
    const int n_tasks     = divceil(p.in_size, p.n_work_items * REGS);
    const int n_tasks_cpu = n_tasks * p.alpha;
    const int n_tasks_gpu = n_tasks - n_tasks_cpu;
    const int n_flags     = n_tasks + 1;
#ifdef OCL_2_0
    T *              h_in_out = (T *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, p.in_size * sizeof(T), 0);
    T *              d_in_out = h_in_out;
    std::atomic_int *h_flags  = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, n_flags * sizeof(std::atomic_int), 0);
    std::atomic_int *d_flags  = h_flags;
    std::atomic_int *worklist = (std::atomic_int *)clSVMAlloc(
        ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(std::atomic_int), 0);
    ALLOC_ERR(worklist);
#else
    T *    h_in_out = (T *)malloc(n_tasks * p.n_work_items * REGS * sizeof(T));
    cl_mem d_in_out = clCreateBuffer(
        ocl.clContext, CL_MEM_READ_WRITE, n_tasks_gpu * p.n_work_items * REGS * sizeof(T), NULL, &clStatus);
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    cl_mem           d_flags = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, n_flags * sizeof(int), NULL, &clStatus);
    CL_ERR();
#endif
    T *h_in_backup = (T *)malloc(p.in_size * sizeof(T)); // Este é o gold
    ALLOC_ERR(h_in_out, h_flags, h_in_backup);
    clFinish(ocl.clCommandQueue);
    //timer.stop("Allocation");
    //timer.print("Allocation", 1);

    // Initialize
    //timer.start("Initialization");
    const int max_wi = ocl.max_work_items(ocl.clKernel);
    //new_read_input(h_in_out, p);
	read_input(h_in_out, p);	
#ifdef OCL_2_0
    h_flags[0].store(1);
#else
    h_flags[0]           = 1;
    h_flags[n_tasks_cpu] = 1;
#endif
    //timer.stop("Initialization");
    //timer.print("Initialization", 1);

// Ler gold
// Para gerar novamente o gold, modificar o arquivo atual de gold para agold... e colocar ele no path e descomentar laa embaixo a função create gold
// *********************** Lendo GOLD   *****************************
	update_timestamp();
    char filename[300];
    FILE *finput;

   snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/SC/gold_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
    if (finput = fopen(filename, "rb")) {
        fread(h_in_backup, p.in_size * sizeof(T), 1 , finput);
    } else {
        printf("Error reading gold file\n");
        exit(1);
    }
	fclose(finput);	

//*********************************************************************
    //memcpy(h_in_backup, h_in_out, p.in_size * sizeof(T)); // Backup for reuse across iterations

    // Loop over main kernel
    for(int rep = 0; rep < p.n_reps; rep++) {
		//printf("Estou na rep %d\n",rep);
		update_timestamp();
#ifndef OCL_2_0
    // Copy to device
   // timer.start("Copy To Device");
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_in_out, CL_TRUE, 0, n_tasks_gpu * p.n_work_items * REGS * sizeof(T),
        h_in_out + n_tasks_cpu * p.n_work_items * REGS, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(
        ocl.clCommandQueue, d_flags, CL_TRUE, 0, n_flags * sizeof(int), h_flags, 0, NULL, NULL);
    CL_ERR();
    clFinish(ocl.clCommandQueue);
    //timer.stop("Copy To Device");
    //timer.print("Copy To Device", 1);
#endif
        // Reset
        //memcpy(h_in_out, h_in_backup, p.in_size * sizeof(T));
        memset(h_flags, 0, n_flags * sizeof(atomic_int));
#ifdef OCL_2_0
        h_flags[0].store(1);
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#else
        h_flags[0]           = 1;
        h_flags[n_tasks_cpu] = 1;
        clStatus = clEnqueueWriteBuffer(
            ocl.clCommandQueue, d_in_out, CL_TRUE, 0, n_tasks_gpu * p.n_work_items * REGS * sizeof(T),
            h_in_out + n_tasks_cpu * p.n_work_items * REGS, 0, NULL, NULL);
	update_timestamp();
        clStatus = clEnqueueWriteBuffer(
            ocl.clCommandQueue, d_flags, CL_TRUE, 0, n_flags * sizeof(int), h_flags, 0, NULL, NULL);
        CL_ERR();
        clFinish(ocl.clCommandQueue);
#endif

    //   if(rep >= p.n_warmup)
       //     timer.start("Kernel");

        clSetKernelArg(ocl.clKernel, 0, sizeof(int), &p.in_size);
        clSetKernelArg(ocl.clKernel, 1, sizeof(int), &p.remove_value);
        clSetKernelArg(ocl.clKernel, 2, p.n_work_items * sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 3, sizeof(int), NULL);
        clSetKernelArg(ocl.clKernel, 4, sizeof(int), &n_tasks);
        clSetKernelArg(ocl.clKernel, 5, sizeof(float), &p.alpha);
#ifdef OCL_2_0
        clSetKernelArgSVMPointer(ocl.clKernel, 6, d_in_out);
        clSetKernelArgSVMPointer(ocl.clKernel, 7, d_in_out);
        clSetKernelArgSVMPointer(ocl.clKernel, 8, d_flags);
        clSetKernelArgSVMPointer(ocl.clKernel, 9, worklist);
        clSetKernelArg(ocl.clKernel, 10, sizeof(int), NULL);
#else
        clSetKernelArg(ocl.clKernel, 6, sizeof(cl_mem), &d_in_out);
        clSetKernelArg(ocl.clKernel, 7, sizeof(cl_mem), &d_in_out);
        clSetKernelArg(ocl.clKernel, 8, sizeof(cl_mem), &d_flags);
#endif
		update_timestamp();
        // Kernel launch
        size_t ls[1] = {(size_t)p.n_work_items};
        size_t gs[1] = {(size_t)p.n_work_items * p.n_work_groups};
        if(gs[0] > 0) {
            assert(ls[0] <= max_wi && 
                "The work-group size is greater than the maximum work-group size that can be used to execute this kernel");
#ifdef LOGS
        start_iteration();
#endif
	update_timestamp();		
            clStatus = clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel, 1, NULL, gs, ls, 0, NULL, NULL);
            CL_ERR();
        }
	update_timestamp();
        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.in_size, p.remove_value, p.n_threads,
            p.n_work_items, n_tasks, p.alpha
#ifdef OCL_2_0
            ,
            worklist
#endif
            );
update_timestamp();
        clFinish(ocl.clCommandQueue);
        main_thread.join();

//        if(rep >= p.n_warmup)
        //    timer.stop("Kernel");
   // }
    //timer.print("Kernel", p.n_reps);
#ifdef LOGS
        end_iteration();
#endif

#ifndef OCL_2_0
    // Copy back
	update_timestamp();
  //  timer.start("Copy Back and Merge");
    if(p.alpha < 1.0) {
        int offset = n_tasks_cpu == 0 ? 1 : 2;
        clStatus   = clEnqueueReadBuffer(ocl.clCommandQueue, d_in_out, CL_TRUE, 0,
            n_tasks_gpu * p.n_work_items * REGS * sizeof(T), h_in_out + h_flags[n_tasks_cpu] - offset, 0, NULL,
            NULL);
        CL_ERR();
    }
    clFinish(ocl.clCommandQueue);
    //timer.stop("Copy Back and Merge");
    //timer.print("Copy Back and Merge", 1);
#endif

create_output(h_in_backup,p);

//******************************************************
/*
	int param1 = divceil(p.in_size, p.n_work_items * REGS);
	int param2 = param1 * p.n_work_items * REGS * sizeof(T);
	//printf("Criando Input\n");
   char filename5[100];
    snprintf(filename5, 100, "agold_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
    if (finput = fopen(filename5, "wb")) {
        fwrite(h_in_out, param2, 1 , finput);
    } else {
        printf("Error writing input file");
        exit(1);
    }
    fclose(finput);
*/
	//printf("Criei Input\n");
//*****************************************************

    // Verify answer
   // verify(h_in_out, h_in_backup, p.in_size, p.remove_value, (p.in_size * p.compaction_factor) / 100);
	//printf("Verificando Input\n");

   // timer.start("Compare");
	update_timestamp();
    err = new_compare_output(h_in_out, h_in_backup, (p.in_size * p.compaction_factor) / 100);
    //timer.stop("Compare");
	update_timestamp();

	//printf("Terminei de Verificar\n");
// Aqui ver se houve erros 
        if(err > 0) {
      //      printf("Errors: %d\n",err);
			snprintf(filename, 300, "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/SC/gold_%d_%d_%d",p.in_size,p.n_work_items,p.compaction_factor); // Gold com a resolução 
			if (finput = fopen(filename, "rb")) {
				fread(h_in_backup,p.in_size * sizeof(T), 1 , finput);
			} else {
				printf("Error reading gold file\n");
				exit(1);
			}
			fclose(finput);	
        } else {
			update_timestamp();
            printf(".");
        }
		update_timestamp();
    	//new_read_input(h_in_out, p);
		read_input(h_in_out, p);
#ifdef LOGS
        log_error_count(err);
#endif
	sleep(3);
	}
#ifdef LOGS
    end_log_file();
#endif

	//timer.print("Compare",p.n_reps);
	//timer.print("Kernel",p.n_reps);
    // Free memory
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in_out);
    clSVMFree(ocl.clContext, h_flags);
    clSVMFree(ocl.clContext, worklist);
#else
    free(h_in_out);
    free(h_flags);
    clStatus = clReleaseMemObject(d_in_out);
    clStatus = clReleaseMemObject(d_flags);
    CL_ERR();
#endif
    free(h_in_backup);
    ocl.release();
    timer.stop("Deallocation");
   // timer.print("Deallocation", 1);

    //printf("Test Passed\n");
    return 0;
}
