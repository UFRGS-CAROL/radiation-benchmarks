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
#include "support/partitioner.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>
#include <stdio.h>
#include <string.h>

//*****************************************  LOG  ***********************************//
#ifdef LOGS
#include "log_helper.h"
#endif
//************************************************************************************//

// Params ---------------------------------------------------------------------
struct Params {

    int         platform;
    int         device;
    int         n_work_items;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    const char *comparison_file;
    int         display = 0;
	int 		loop; 

    Params(int argc, char **argv) {
/*
//CPU
        platform        = 0;
        device          = 0;
        n_work_items    = 16;
        n_threads       = 4;
        n_warmup        = 10;
        n_reps          = 1100;
        alpha           = 1.0;
		loop 			= 1;
        file_name       = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/";
        comparison_file = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/";
*/
/*

//GPU
        platform        = 0;
        device          = 0;
        n_work_items    = 16;
        n_threads       = 4;
        n_warmup        = 10;
        n_reps          = 1100;
        alpha           = 0.0;
	loop 			= 1;
        file_name       = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/";
        comparison_file = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/";
*/

//CPU +GPU
        platform        = 0;
        device          = 0;
        n_work_items    = 16;
        n_threads       = 4;
        n_warmup        = 10;
        n_reps          = 1100;
        alpha           = 0.1;
		loop 			= 1;
        file_name       = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/input/urban_input/";
        comparison_file = "/home/carol/radiation-benchmarks/src/opencl/Heterogenous_GPU_CPU/CEDD/output/urban_output/";


        int opt;
        while((opt = getopt(argc, argv, "hp:d:i:t:w:r:a:f:c:xl:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'p': platform        = atoi(optarg); break;
            case 'd': device          = atoi(optarg); break;
            case 'i': n_work_items    = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'a': alpha           = atof(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            case 'x': display         = 1; break;
            case 'l': loop 			  = atoi(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        if(alpha == 0.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_work_items > 0 && "Invalid # of device work-items!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
            assert((n_work_items > 0 || n_threads > 0) && "Invalid # of host + device workers!");
        }
#ifndef CHAI_OPENCV
        assert(display != 1 && "Compile with CHAI_OPENCV");
#endif
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./cedd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -p <P>    OpenCL platform ID (default=0)"
                "\n    -d <D>    OpenCL device ID (default=0)"
                "\n    -i <I>    # of device work-items (default=16)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=10)"
                "\n    -r <R>    # of timed repetition iterations (default=100)"
                "\n    -l <R>    # radiation loop iterations (default=100)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
                "\n              NOTE: Dynamic partitioning used when <A> is not between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    folder containing input video files (default=input/peppa/)"
                "\n    -c <C>    folder containing comparison files (default=output/peppa/)"
                "\n    -x        display output video (with CHAI_OPENCV)"
                "\n");
    }
};
// Eh essa aqui a função utilizada para compare Maio 2018
inline int newest_compare_output(unsigned char **all_out_frames, int image_size,unsigned char **gold, int num_frames, int rowsc, int colsc, int rowsc_, int colsc_) {

   // printf("Entrei compara\n");
    int count_error = 0;
	int i = 0 ;
	int r = 0 ;
	int c = 0 ;
# pragma omp parallel for reduction(+:count_error) private(i,c,r)
    for(int i = 0; i < num_frames; i++) {
		//update_timestamp();
        for(int r = 0; r < rowsc; r++) {
            for(int c = 0; c < colsc; c++) {
                int pix;		
		pix = (int)gold[i][r*colsc+c];
                if((int)all_out_frames[i][r*colsc+c] != pix) {
                    if(r > 3 && r < rowsc-32 && c > 3 && c < colsc-32){
                        count_error++;
#ifdef LOGS
			//printf("Erro frame %d aqui: %d\n",i,colsc);
		        char error_detail[250];
        		sprintf(error_detail,"p: [%d, %d],r: %d,e: %d,Image Size:%d, #Frame:%d, #TOTFRA:%d",r,c,(int)all_out_frames[i][r*colsc+c],pix,image_size,i, num_frames);

       			log_error_detail(error_detail);
#endif
                    }
                }
            }
        }
    }
    update_timestamp();
    if((float)count_error / (float)(image_size * num_frames) >= 1e-6){
        printf("Test failed\n");
        //exit(EXIT_FAILURE);
    }
	//printf("Sai compare\n");
    return count_error;
}


inline int new_compare_output(unsigned char **all_out_frames, int image_size, const char *file_name, int num_frames, int rowsc, int colsc, int rowsc_, int colsc_) {

    //printf("Entrei compara sem memoria\n");
    int count_error = 0;
    int new_counter = 0;
# pragma omp parallel for
    for(int i = 0; i < num_frames; i++) {
    update_timestamp();
        // Compare to output file
        char FileName[300];
        sprintf(FileName, "%s%d.txt", file_name, i);
        FILE *out_file = fopen(FileName, "r");
        if(!out_file) {
            printf("Error Reading output file\n");
            return 1;
        }
#if PRINT
        printf("Reading Output: %s\n", file_name);
#endif

        for(int r = 0; r < rowsc; r++) {
            for(int c = 0; c < colsc; c++) {
                int pix;
                fscanf(out_file, "%d ", &pix);
                if((int)all_out_frames[i][r*colsc+c] != pix) {
			new_counter++;
                    if(r > 3 && r < rowsc-32 && c > 3 && c < colsc-32){


                        count_error++;
#ifdef LOGS
		        char error_detail[250];
        		sprintf(error_detail,"p: [%d, %d],r: %d,e: %d,Image Size:%d, #Frame:%d, #TOTFRA:%d",r,c,(int)all_out_frames[i][r*colsc+c],pix,image_size,i, num_frames);

       			 log_error_detail(error_detail);
#endif

                    }
                }
            }
            // Scan until end of row
            if(colsc<colsc_) fscanf(out_file, "%*[^\n]\n");
        }
        // Scan until end of frame
        for(int rr=rowsc;rr<rowsc_;rr++) fscanf(out_file, "%*[^\n]\n");

        fclose(out_file);
    }

    if((float)count_error / (float)(image_size * num_frames) >= 1e-6){
        printf("Test failed\n");
        //exit(EXIT_FAILURE);
    }
	//printf("Sai compare\n");
    return count_error;
}

// Input Data -----------------------------------------------------------------
void new_read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {
	//printf("Lendo Input\n");
    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[300];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
        //all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
        fclose(fp);
    }
}
void new_read_gold(unsigned char** all_gray_frames, int rowsc, int colsc, int in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[300];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

//        fscanf(fp, "%d\n", &rowsc);
//        fscanf(fp, "%d\n", &colsc);

//        in_size = rowsc * colsc * sizeof(unsigned char);
        //all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
        fclose(fp);
    }
}

void read_gold(unsigned char** all_gray_frames, int rowsc, int colsc, int in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[300];
        sprintf(FileName, "%s%d.txt", p.comparison_file, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

      //  fscanf(fp, "%d\n", &rowsc);
      // fscanf(fp, "%d\n", &colsc);

//        in_size = rowsc * colsc * sizeof(unsigned char);
        all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                //fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
                fscanf(fp, "%d ",&all_gray_frames[task_id][i * colsc + j]);
				
            }
        }
/*
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {

			
				printf("%d \t",all_gray_frames[task_id][i * colsc + j]);			
	//          fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
			
        }
		printf("\n");
        }
*/
        fclose(fp);
//printf("\n");
    }
}
void read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[300];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
//	printf("In size%d: %d \t\n",task_id,in_size);
        all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
/*
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {

		printf("%u \t",all_gray_frames[task_id][i * colsc + j]);			
//                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
		printf("\n");
        }
*/	
        fclose(fp);
	//printf("\n");
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    Params      p(argc, argv);
    OpenCLSetup ocl(p.platform, p.device);
    cl_int      clStatus;
    Timer       timer;

	int err = 0;

printf("-p %d -d %d -i %d -a %.2f -t %d \n",p.platform , p.device, p.n_work_items,p.alpha,p.n_threads);

#ifdef LOGS
    set_iter_interval_print(10);
    char test_info[300];
    snprintf(test_info, 300, "-i %d -a %.2f -t %d -r %d -f %s ",p.n_work_items,p.alpha,p.n_threads,p.n_warmup+p.n_reps,p.file_name);
    start_log_file("openclCannyEdgeDetection", test_info);
	//printf("Com LOG\n");
#endif


    // Initialize (part 1)
    timer.start("Initialization");
    const int max_wi_gauss  = ocl.max_work_items(ocl.clKernel_gauss);
    const int max_wi_sobel  = ocl.max_work_items(ocl.clKernel_sobel);
    const int max_wi_nonmax = ocl.max_work_items(ocl.clKernel_nonmax);
    const int max_wi_hyst   = ocl.max_work_items(ocl.clKernel_hyst);
    const int n_frames = p.n_warmup + p.n_reps;
    unsigned char **all_gray_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
	//update_timestamp();
//******************************* Alocando Memoria para o Gold *****************************
    unsigned char **gold = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
//*****************************************************************************************
    int     rowsc, colsc, in_size;
	update_timestamp();
    read_input(all_gray_frames, rowsc, colsc, in_size, p);
//******************************* Lendo Gold *********************************************
	update_timestamp();    
	read_gold(gold, rowsc, colsc, in_size, p);
//****************************************************************************************
    timer.stop("Initialization");
	update_timestamp();
    // Allocate buffers
    timer.start("Allocation");
    const int CPU_PROXY = 0;
    const int GPU_PROXY = 1;
    unsigned char *    h_in_out[2];
    h_in_out[CPU_PROXY] = (unsigned char *)malloc(in_size);
#ifdef OCL_2_0
    h_in_out[GPU_PROXY]     = (unsigned char *)clSVMAlloc(ocl.clContext, CL_MEM_SVM_FINE_GRAIN_BUFFER, in_size, 0);
    unsigned char *d_in_out = h_in_out[GPU_PROXY];
#else
    h_in_out[GPU_PROXY] = (unsigned char *)malloc(in_size);
    cl_mem d_in_out     = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
#endif
    unsigned char *h_interm_cpu_proxy = (unsigned char *)malloc(in_size);
    unsigned char *h_theta_cpu_proxy  = (unsigned char *)malloc(in_size);
    cl_mem         d_interm_gpu_proxy = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
    cl_mem         d_theta_gpu_proxy  = clCreateBuffer(ocl.clContext, CL_MEM_READ_WRITE, in_size, NULL, &clStatus);
    std::atomic<int> next_frame;
    clFinish(ocl.clCommandQueue);
    ALLOC_ERR(h_in_out[CPU_PROXY], h_in_out[GPU_PROXY], h_interm_cpu_proxy, h_theta_cpu_proxy);
    CL_ERR();
    timer.stop("Allocation");
    //timer.print("Allocation", 1);

    // Initialize (part 2)
    timer.start("Initialization");
    unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    for(int i = 0; i < n_frames; i++) {
        all_out_frames[i] = (unsigned char *)malloc(in_size);
    }
    std::atomic_int *worklist    = (std::atomic_int *)malloc(sizeof(std::atomic_int));
    ALLOC_ERR(worklist);
    if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
        worklist[0].store(0);
    }
    next_frame.store(0);
    timer.stop("Initialization");
    //timer.print("Initialization", 1);
    //timer.start("Total Proxies");

for(int rep = 0; rep < p.loop; rep++) {
    update_timestamp();
    //printf("Rep:%d\n",rep);
    timer.start("Total Proxies");
    CoarseGrainPartitioner partitioner = partitioner_create(n_frames, p.alpha, worklist);
    std::vector<std::thread> proxy_threads;
#ifdef LOGS
        start_iteration();
#endif
	//printf("Nova it\n");
	timer.start("TESTE");
    for(int proxy_tid = 0; proxy_tid < 2; proxy_tid++) {
        proxy_threads.push_back(std::thread([&, proxy_tid]() {

            if(proxy_tid == GPU_PROXY) {
//printf("GPU\n");
                for(int task_id = gpu_first(&partitioner); gpu_more(&partitioner); task_id = gpu_next(&partitioner)) {
    				update_timestamp();
                    // Next frame
                    memcpy(h_in_out[proxy_tid], all_gray_frames[task_id], in_size);

#ifndef OCL_2_0
                    // Copy to Device
                    timer.start("GPU Proxy: Copy To Device");
                    clStatus = clEnqueueWriteBuffer(
                        ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size, h_in_out[proxy_tid], 0, NULL, NULL);
                    CL_ERR();
                    clFinish(ocl.clCommandQueue);
                    timer.stop("GPU Proxy: Copy To Device");
#endif

                    timer.start("GPU Proxy: Kernel");
                    // Execution configuration
                    size_t ls[2]     = {(size_t)p.n_work_items, (size_t)p.n_work_items};
                    size_t gs[2]     = {(size_t)(colsc - 2), (size_t)(rowsc - 2)};
                    size_t offset[2] = {(size_t)1, (size_t)1};

                    // GAUSSIAN KERNEL
                    // Set arguments
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_gauss, 0, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_gauss, 0, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_gauss, 1, sizeof(cl_mem), &d_interm_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_gauss, 2, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_gauss, 3, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_gauss, 4, (p.n_work_items + 2) * (p.n_work_items + 2) * sizeof(int), NULL);
                    assert(ls[0]*ls[1] <= max_wi_gauss && 
                        "The work-group size is greater than the maximum work-group size that can be used to execute gaussian kernel");
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_gauss, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

                    // SOBEL KERNEL
                    // Set arguments
                    clSetKernelArg(ocl.clKernel_sobel, 0, sizeof(cl_mem), &d_interm_gpu_proxy);
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_sobel, 1, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_sobel, 1, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_sobel, 2, sizeof(cl_mem), &d_theta_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_sobel, 3, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_sobel, 4, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_sobel, 5, (p.n_work_items + 2) * (p.n_work_items + 2) * sizeof(int), NULL);
                    assert(ls[0]*ls[1] <= max_wi_sobel && 
                        "The work-group size is greater than the maximum work-group size that can be used to execute sobel kernel");
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_sobel, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();
			update_timestamp();
                    // NON-MAXIMUM SUPPRESSION KERNEL
                    // Set arguments
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_nonmax, 0, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_nonmax, 0, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_nonmax, 1, sizeof(cl_mem), &d_interm_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_nonmax, 2, sizeof(cl_mem), &d_theta_gpu_proxy);
                    clSetKernelArg(ocl.clKernel_nonmax, 3, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_nonmax, 4, sizeof(int), &colsc);
                    clSetKernelArg(ocl.clKernel_nonmax, 5, (p.n_work_items + 2) * (p.n_work_items + 2) * sizeof(int), NULL);
                    assert(ls[0]*ls[1] <= max_wi_nonmax && 
                        "The work-group size is greater than the maximum work-group size that can be used to execute non-maximum suppression kernel");
                    // Kernel launch
                    clStatus = clEnqueueNDRangeKernel(
                        ocl.clCommandQueue, ocl.clKernel_nonmax, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

                    // HYSTERESIS KERNEL
                    // Set arguments
                    clSetKernelArg(ocl.clKernel_hyst, 0, sizeof(cl_mem), &d_interm_gpu_proxy);
#ifdef OCL_2_0
                    clSetKernelArgSVMPointer(ocl.clKernel_hyst, 1, d_in_out);
#else
                    clSetKernelArg(ocl.clKernel_hyst, 1, sizeof(cl_mem), &d_in_out);
#endif
                    clSetKernelArg(ocl.clKernel_hyst, 2, sizeof(int), &rowsc);
                    clSetKernelArg(ocl.clKernel_hyst, 3, sizeof(int), &colsc);
                    assert(ls[0]*ls[1] <= max_wi_hyst && 
                        "The work-group size is greater than the maximum work-group size that can be used to execute hysteresis kernel");
                    // Kernel launch
                    clStatus =
                        clEnqueueNDRangeKernel(ocl.clCommandQueue, ocl.clKernel_hyst, 2, offset, gs, ls, 0, NULL, NULL);
                    CL_ERR();

                    clFinish(ocl.clCommandQueue);
                    timer.stop("GPU Proxy: Kernel");

#ifndef OCL_2_0
                    timer.start("GPU Proxy: Copy Back");
                    clStatus = clEnqueueReadBuffer(
                        ocl.clCommandQueue, d_in_out, CL_TRUE, 0, in_size, h_in_out[proxy_tid], 0, NULL, NULL);
                    CL_ERR();
                    clFinish(ocl.clCommandQueue);
                    timer.stop("GPU Proxy: Copy Back");
#endif

                    memcpy(all_out_frames[task_id], h_in_out[proxy_tid], in_size);
                    
                }

            } else if(proxy_tid == CPU_PROXY) {
//printf("CPU\n");
                for(int task_id = cpu_first(&partitioner); cpu_more(&partitioner); task_id = cpu_next(&partitioner)) {
    				update_timestamp();
                    // Next frame
                    memcpy(h_in_out[proxy_tid], all_gray_frames[task_id], in_size);

                    // Launch CPU threads
                    timer.start("CPU Proxy: Kernel");
                    std::thread main_thread(run_cpu_threads, h_in_out[proxy_tid], h_interm_cpu_proxy, h_theta_cpu_proxy,
                        rowsc, colsc, p.n_threads, task_id);
	
                    main_thread.join();
                    timer.stop("CPU Proxy: Kernel");

                    memcpy(all_out_frames[task_id], h_in_out[proxy_tid], in_size);

                }

            }

        }));
    }
	update_timestamp();
    std::for_each(proxy_threads.begin(), proxy_threads.end(), [](std::thread &t) { t.join(); });
    clFinish(ocl.clCommandQueue);
#ifdef LOGS
        end_iteration();
#endif
	timer.stop("TESTE");
    timer.stop("Total Proxies");
    //timer.print("Total Proxies", 1);
    //printf("CPU Proxy:\n");
    //printf("\t");
    //timer.print("CPU Proxy: Kernel", 1);
    //printf("GPU Proxy:\n");
    //printf("\t");
    //timer.print("GPU Proxy: Copy To Device", 1);
    //printf("\t");
    //timer.print("GPU Proxy: Kernel", 1);
    //printf("\t");
    //timer.print("GPU Proxy: Copy Back", 1);

#ifdef CHAI_OPENCV
    // Display the result
    if(p.display){
        for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
            cv::Mat out_frame = cv::Mat(rowsc, colsc, CV_8UC1);
            memcpy(out_frame.data, all_out_frames[rep], in_size);
            if(!out_frame.empty())
                imshow("canny", out_frame);
            if(cv::waitKey(30) >= 0)
                break;
        }
    }
#endif

    // Verify answer
	update_timestamp();

//verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);
/*
printf("Imprimindo a Saida do Programa\n");
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
		printf("%d \t",all_gray_frames[0][i * colsc + j]);			
            }
		printf("\n");
        }
*/

//err = new_compare_output(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);
	timer.start("Tempo Compare");
	err = newest_compare_output(all_out_frames, in_size, gold, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);
	timer.stop("Tempo Compare");
// Aqui ver se houve erros 
        if(err > 0) {
            printf("Errors: %d\n",err);
			update_timestamp();
		    new_read_input(all_gray_frames, rowsc, colsc, in_size, p);
			update_timestamp();
		    //new_read_gold(gold, rowsc, colsc, in_size, p);		
			 
        } else {
            printf(".");
        }
   // new_read_input(all_gray_frames, rowsc, colsc, in_size, p);
#ifdef LOGS
        log_error_count(err);
#endif
//printf("Acabei uma it\n");
}

#ifdef LOGS
    end_log_file();
#endif

	timer.print("TESTE",p.loop);
	timer.print("Tempo Compare",p.loop);
/*
    timer.print("Total Proxies",  p.loop);
    printf("CPU Proxy:\n");
    printf("\t");
    timer.print("CPU Proxy: Kernel",  p.loop);
    printf("GPU Proxy:\n");
    printf("\t");
    timer.print("GPU Proxy: Copy To Device",  p.loop);
    printf("\t");
    timer.print("GPU Proxy: Kernel",  p.loop);
    printf("\t");
    timer.print("GPU Proxy: Copy Back",  p.loop);
*/

    // Release buffers
    timer.start("Deallocation");
#ifdef OCL_2_0
    clSVMFree(ocl.clContext, h_in_out[GPU_PROXY]);
#else
    free(h_in_out[GPU_PROXY]);
    clStatus = clReleaseMemObject(d_in_out);
#endif
    free(h_in_out[CPU_PROXY]);
    free(h_interm_cpu_proxy);
    free(h_theta_cpu_proxy);
    for(int i = 0; i < n_frames; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < n_frames; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);
    clStatus = clReleaseMemObject(d_interm_gpu_proxy);
    clStatus = clReleaseMemObject(d_theta_gpu_proxy);
    CL_ERR();
    free(worklist);
    ocl.release();
    timer.stop("Deallocation");
    //timer.print("Deallocation", 1);

    //printf("Test Passed\n");
    return 0;
}
