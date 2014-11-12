#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <fstream>
#include <vector>

#include <sys/time.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <CL/cl_ext.h>

#include "sort.h"

#define FPTYPE uint

#define ITERACTIONS 1		 //loop

using namespace std;

int gpu_work = 8; // default value is 8 (100% GPU, 0% CPU), max value is also 8
int probSizes[7] = { 1, 8, 32, 64 , 128, 256, 512}; // Problem Sizes
int size = probSizes[3];
FPTYPE bytes;

FPTYPE *h_idata; // input data

void runTest(int test_number) {

    int err;


    const int radix_width = 4;
    const int num_digits = (int)pow((double)2, radix_width);

    const size_t local_wsize  = 256;
    const size_t global_wsize = 16384;
    const size_t num_work_groups = global_wsize / local_wsize;

    // Allocate host memory for output data
    FPTYPE *h_odata = (FPTYPE*)malloc(bytes);
    if(h_odata == NULL){
        printf("Errors allocating host memory for output data\n");
        exit(1);
    }


    // Allocate GPU device memory
    cl_mem d_idata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem d_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem d_isums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    num_work_groups * num_digits * sizeof(FPTYPE), NULL, &err);
    CL_CHECK_ERROR(err);


    // Allocate CPU device memory
    cl_mem cpu_d_idata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem cpu_d_odata = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    CL_CHECK_ERROR(err);
    cl_mem cpu_d_isums = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        num_work_groups * num_digits * sizeof(FPTYPE), NULL, &err);
    CL_CHECK_ERROR(err);





    // Set the kernel arguments for the reduction kernel
    err = clSetKernelArg(reduce, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 1, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 2, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(reduce, 3, local_wsize * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem), (void*)&cpu_d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_reduce, 1, sizeof(cl_mem), (void*)&cpu_d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_reduce, 2, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_reduce, 3, local_wsize * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the top-level scan
    err = clSetKernelArg(top_scan, 0, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(top_scan, 2, local_wsize * 2 * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_top_scan, 0, sizeof(cl_mem), (void*)&cpu_d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_top_scan, 1, sizeof(cl_int), (void*)&num_work_groups);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_top_scan, 2, local_wsize * 2 * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);

    // Set the kernel arguments for the bottom-level scan
    err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem), (void*)&d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 1, sizeof(cl_mem), (void*)&d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem), (void*)&d_odata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 3, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(bottom_scan, 4, local_wsize * 2 * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem), (void*)&cpu_d_idata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_bottom_scan, 1, sizeof(cl_mem), (void*)&cpu_d_isums);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem), (void*)&cpu_d_odata);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_bottom_scan, 3, sizeof(cl_int), (void*)&size);
    CL_CHECK_ERROR(err);
    err = clSetKernelArg(cpu_bottom_scan, 4, local_wsize * 2 * sizeof(FPTYPE), NULL);
    CL_CHECK_ERROR(err);


    long long time0, time1;
    double gpu_kernel_time = 0, cpu_kernel_time = 0;
    int shift = 0;
    bool even = true;

    ///////////
    // GPU part
    ///////////
    if(gpu_work > 0) {

        // Copy data to GPU
        err = clEnqueueWriteBuffer(command_queue[0], d_idata, true, 0, bytes, h_idata, 0,
                                   NULL, NULL);
        CL_CHECK_ERROR(err);
        err = clFinish(command_queue[0]);
        CL_CHECK_ERROR(err);

        ///////////
        // GPU run
        ///////////
        time0 = get_time();
        for (shift = 0; shift < sizeof(FPTYPE)*gpu_work; shift += radix_width) {

            err = clSetKernelArg(reduce, 4, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);
            err = clSetKernelArg(bottom_scan, 5, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);

            even = ((shift / radix_width) % 2 == 0) ? true : false;

            if (even) {

                err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
                                     (void*)&d_idata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
                                     (void*)&d_idata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
                                     (void*)&d_odata);
                CL_CHECK_ERROR(err);
            }
            else {

                err = clSetKernelArg(reduce, 0, sizeof(cl_mem),
                                     (void*)&d_odata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 0, sizeof(cl_mem),
                                     (void*)&d_odata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(bottom_scan, 2, sizeof(cl_mem),
                                     (void*)&d_idata);
                CL_CHECK_ERROR(err);
            }

            err = clEnqueueNDRangeKernel(command_queue[0], reduce, 1, NULL,
                                         &global_wsize, &local_wsize, 0, NULL, NULL);
            err = clEnqueueNDRangeKernel(command_queue[0], top_scan, 1, NULL,
                                         &local_wsize, &local_wsize, 0, NULL, NULL);
            err = clEnqueueNDRangeKernel(command_queue[0], bottom_scan, 1, NULL,
                                         &global_wsize, &local_wsize, 0, NULL, NULL);
        }
        err = clFinish(command_queue[0]);
        CL_CHECK_ERROR(err);
        time1 = get_time();
        gpu_kernel_time = (double) (time1-time0) / 1000000;
        if (even) {
            err = clEnqueueReadBuffer(command_queue[0], d_odata, true, 0, bytes, h_odata,
                                      0, NULL, NULL);
        } else {
            err = clEnqueueReadBuffer(command_queue[0], d_idata, true, 0, bytes, h_odata,
                                      0, NULL, NULL);
        }
        CL_CHECK_ERROR(err);
        err = clFinish(command_queue[0]);
        CL_CHECK_ERROR(err);
    }

    ///////////
    // CPU part
    ///////////
    if(gpu_work < 8) {

        // Copy data to CPU device
        if (!even && gpu_work > 0) {
            err = clEnqueueWriteBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_odata, 0, NULL, NULL);
        } else if (!even && gpu_work == 0) {
            err = clEnqueueWriteBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_idata, 0, NULL, NULL);

        } else if (even && gpu_work > 0) {
            err = clEnqueueWriteBuffer(command_queue[1], cpu_d_odata, true, 0, bytes, h_odata, 0, NULL, NULL);
        } else {
            err = clEnqueueWriteBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_idata, 0, NULL, NULL);
        }
        CL_CHECK_ERROR(err);
        err = clFinish(command_queue[1]);
        CL_CHECK_ERROR(err);


        ///////////
        // CPU run
        ///////////
        time0 = get_time();
        for (; shift < sizeof(FPTYPE)*8; shift += radix_width) {
 
            err = clSetKernelArg(cpu_reduce, 4, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);

            err = clSetKernelArg(cpu_bottom_scan, 5, sizeof(cl_int), (void*)&shift);
            CL_CHECK_ERROR(err);

            even = ((shift / radix_width) % 2 == 0) ? true : false;

            if (even) {

                err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem),
                                     (void*)&cpu_d_idata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem),
                                     (void*)&cpu_d_idata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem),
                                     (void*)&cpu_d_odata);
                CL_CHECK_ERROR(err);
            }
            else {

                err = clSetKernelArg(cpu_reduce, 0, sizeof(cl_mem),
                                     (void*)&cpu_d_odata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(cpu_bottom_scan, 0, sizeof(cl_mem),
                                     (void*)&cpu_d_odata);
                CL_CHECK_ERROR(err);
                err = clSetKernelArg(cpu_bottom_scan, 2, sizeof(cl_mem),
                                     (void*)&cpu_d_idata);
                CL_CHECK_ERROR(err);
            }

            err = clEnqueueNDRangeKernel(command_queue[1], cpu_reduce, 1, NULL,
                                         &global_wsize, &local_wsize, 0, NULL, NULL);
            err = clEnqueueNDRangeKernel(command_queue[1], cpu_top_scan, 1, NULL,
                                         &local_wsize, &local_wsize, 0, NULL, NULL);
            err = clEnqueueNDRangeKernel(command_queue[1], cpu_bottom_scan, 1, NULL,
                                         &global_wsize, &local_wsize, 0, NULL, NULL);
        }

        err = clFinish(command_queue[1]);
        CL_CHECK_ERROR(err);
        time1 = get_time();
        cpu_kernel_time = (double) (time1-time0) / 1000000;

        if (even) {
            err = clEnqueueReadBuffer(command_queue[1], cpu_d_odata, true, 0, bytes, h_odata, 0, NULL, NULL);
        } else {
            err = clEnqueueReadBuffer(command_queue[1], cpu_d_idata, true, 0, bytes, h_odata, 0, NULL, NULL);
        }

        CL_CHECK_ERROR(err);
        err = clFinish(command_queue[1]);
        CL_CHECK_ERROR(err);
    }


    // check output with GOLD
    FILE *fp;
    if( (fp = fopen("output_sort", "rb" )) == 0 ) {
        printf( "error file output_sort was not opened\n");
        return;
    }
    int num_errors = 0;
    int gold;
    int i;
    int order_errors = 0;
    for (i = 0; i < size; i++) {
        fread(&gold, 1, sizeof(FPTYPE), fp);
        if(h_odata[i] != gold) {
            num_errors++;
        }

        if (i < size -1 && h_odata[i] > h_odata[i + 1]) {
            order_errors++;
        }
    }

    fclose(fp);


    printf("\ntest number: %d", test_number);
    printf("\nGPU kernel time: %.12f", gpu_kernel_time);
    printf("\nCPU kernel time: %.12f", cpu_kernel_time);
    printf("\namount of errors: %d", num_errors);
    printf("\nelements with order errors: %d\n", order_errors);

	// Clean up device memory
	err = clReleaseMemObject(d_idata);
	err = clReleaseMemObject(d_odata);
	err = clReleaseMemObject(d_isums);
	err = clReleaseMemObject(cpu_d_idata);
	err = clReleaseMemObject(cpu_d_odata);
	err = clReleaseMemObject(cpu_d_isums);
    free(h_odata);
}



int main(int argc, char** argv) {

    if(argc > 1) {
        gpu_work = atoi(argv[1]);
    }

    printf("gpu_work = %d\n",gpu_work);

    size = (size * 1024 * 1024) / sizeof(FPTYPE);
    bytes = size * sizeof(FPTYPE);

    initOpenCL();

    // Allocate host memory for input data
    h_idata = (FPTYPE*)malloc(bytes);
    if(h_idata == NULL){
        printf("Errors allocating host memory for output data\n");
        exit(1);
    }
    FILE *fp;
    if( (fp = fopen("input_sort", "rb" )) == 0 ){
        printf( "The file input_sort was not opened\n");
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fread(&(h_idata[i]), 1, sizeof(FPTYPE), fp);
    }
    fclose(fp);

    //LOOP START
    int loop;
    for(loop=0; loop<ITERACTIONS; loop++) {
        runTest(loop);
    }
}
