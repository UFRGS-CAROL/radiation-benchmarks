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

#include "support/cuda-setup.h"
#include "kernel.h"
#include "support/common.h"
#include "support/timer.h"
#include "support/verify.h"

#include <unistd.h>
#include <thread>
#include <assert.h>

// Params ---------------------------------------------------------------------
struct Params {

    int         device;
    int         n_gpu_threads;
    int         n_gpu_blocks;
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
        device        = 0;
        n_gpu_threads = 16;
        n_gpu_blocks  = 32;
        n_threads     = 4;
        n_warmup      = 5;
        n_reps        = 50;
        alpha         = 0.1;
        file_name     = "input/control.txt";
        in_size_i = in_size_j = 3;
        out_size_i = out_size_j = 300;
        int opt;
        while((opt = getopt(argc, argv, "hd:i:g:t:w:r:a:f:m:n:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'd': device        = atoi(optarg); break;
            case 'i': n_gpu_threads  = atoi(optarg); break;
            case 'g': n_gpu_blocks = atoi(optarg); break;
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
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
        } else if(alpha == 1.0) {
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else if(alpha > 0.0 && alpha < 1.0) {
            assert(n_gpu_threads > 0 && "Invalid # of device threads!");
            assert(n_gpu_blocks > 0 && "Invalid # of device blocks!");
            assert(n_threads > 0 && "Invalid # of host threads!");
        } else {
#ifdef CUDA_8_0
            assert((n_gpu_threads > 0 && n_gpu_blocks > 0 || n_threads > 0) && "Invalid # of host + device workers!");
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
                "\n    -d <D>    CUDA device ID (default=0)"
                "\n    -i <I>    # of device threads per block (default=16)"
                "\n    -g <G>    # of device blocks (default=32)"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of output elements to process on host (default=0.1)"
#ifdef CUDA_8_0
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
void read_input(XYZ *in, const Params &p) {

    // Open input file
    FILE *f = NULL;
    f       = fopen(p.file_name, "r");
    if(f == NULL) {
        puts("Error opening file");
        exit(-1);
    }

    // Store points from input file to array
    int k = 0, ic = 0;
    XYZ v[10000];
#if DOUBLE_PRECISION
    while(fscanf(f, "%lf,%lf,%lf", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#else
    while(fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#endif
    {
        ic++;
    }
    for(int i = 0; i <= p.in_size_i; i++) {
        for(int j = 0; j <= p.in_size_j; j++) {
            in[i * (p.in_size_j + 1) + j].x = v[k].x;
            in[i * (p.in_size_j + 1) + j].y = v[k].y;
            in[i * (p.in_size_j + 1) + j].z = v[k].z;
            //k++;
            k = (k + 1) % 16;
        }
    }
}

// Main -----------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    CUDASetup    setcuda(p.device);
    Timer        timer;
    cudaError_t  cudaStatus;

    // Allocate
    timer.start("Allocation");
    int in_size   = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
    int out_size  = p.out_size_i * p.out_size_j * sizeof(XYZ);
    int n_tasks_i = divceil(p.out_size_i, p.n_gpu_threads);
    int n_tasks_j = divceil(p.out_size_j, p.n_gpu_threads);
    int n_tasks   = n_tasks_i * n_tasks_j;
#ifdef CUDA_8_0
    XYZ * h_in;
    cudaStatus = cudaMallocManaged(&h_in, in_size);
    XYZ * h_out;
    cudaStatus = cudaMallocManaged(&h_out, out_size);
    XYZ * d_in  = h_in;
    XYZ * d_out = h_out;
    std::atomic_int * worklist;
    cudaStatus = cudaMallocManaged(&worklist, sizeof(std::atomic_int));
#else
    XYZ *  h_in        = (XYZ *)malloc(in_size);
    XYZ *  h_out       = (XYZ *)malloc(out_size);
    XYZ *  h_out_merge = (XYZ *)malloc(out_size);
    XYZ* d_in;
    cudaStatus = cudaMalloc((void**)&d_in, in_size);
    XYZ* d_out;
    cudaStatus = cudaMalloc((void**)&d_out, out_size);
    ALLOC_ERR(h_in, h_out, h_out_merge);
#endif
    CUDA_ERR();
    cudaDeviceSynchronize();
    timer.stop("Allocation");
    timer.print("Allocation", 1);

    // Initialize
    timer.start("Initialization");
    const int max_gpu_threads = setcuda.max_gpu_threads();
    read_input(h_in, p);
    cudaDeviceSynchronize();
    timer.stop("Initialization");
    timer.print("Initialization", 1);

#ifndef CUDA_8_0
    // Copy to device
    timer.start("Copy To Device");
    cudaStatus = cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    CUDA_ERR();
    timer.stop("Copy To Device");
    timer.print("Copy To Device", 1);
#endif

    // Loop over main kernel
    for(int rep = 0; rep < p.n_warmup + p.n_reps; ++rep) {

// Reset
#ifdef CUDA_8_0
        if(p.alpha < 0.0 || p.alpha > 1.0) { // Dynamic partitioning
            worklist[0].store(0);
        }
#endif

        if(rep >= p.n_warmup)
            timer.start("Kernel");

        // Launch GPU threads
        // Kernel launch
        if(p.n_gpu_blocks > 0) {
            assert(p.n_gpu_threads * p.n_gpu_threads <= max_gpu_threads && 
                "The thread block size is greater than the maximum thread block size that can be used on this device");
            cudaStatus = call_Bezier_surface(p.n_gpu_blocks, p.n_gpu_threads, n_tasks, p.alpha,
                p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j, 
                in_size
#ifdef CUDA_8_0
                + sizeof(int)
#endif
                , d_in, d_out
#ifdef CUDA_8_0
                , (int*)worklist
#endif
                );
            CUDA_ERR();
        }

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in, h_out, n_tasks, p.alpha, p.n_threads, p.n_gpu_threads, p.in_size_i,
            p.in_size_j, p.out_size_i, p.out_size_j
#ifdef CUDA_8_0
            , worklist
#endif
            );

        cudaDeviceSynchronize();
        main_thread.join();

        if(rep >= p.n_warmup)
            timer.stop("Kernel");
    }
    timer.print("Kernel", p.n_reps);

#ifndef CUDA_8_0
    // Copy back
    timer.start("Copy Back and Merge");
    cudaStatus = cudaMemcpy(h_out_merge, d_out, out_size, cudaMemcpyDeviceToHost);
    CUDA_ERR();
    cudaDeviceSynchronize();
    // Merge
    int cut = n_tasks * p.alpha;
    for(unsigned int t = 0; t < cut; ++t) {
        const int ty  = t / n_tasks_j;
        const int tx  = t % n_tasks_j;
        int       row = ty * p.n_gpu_threads;
        int       col = tx * p.n_gpu_threads;
        for(int i = row; i < row + p.n_gpu_threads; ++i) {
            for(int j = col; j < col + p.n_gpu_threads; ++j) {
                if(i < p.out_size_i && j < p.out_size_j) {
                    h_out_merge[i * p.out_size_j + j] = h_out[i * p.out_size_j + j];
                }
            }
        }
    }
    timer.stop("Copy Back and Merge");
    timer.print("Copy Back and Merge", 1);
#endif

// Verify answer
#ifdef CUDA_8_0
    verify(h_in, h_out, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);
#else
    verify(h_in, h_out_merge, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j);
#endif

    // Free memory
    timer.start("Deallocation");
#ifdef CUDA_8_0
    cudaStatus = cudaFree(h_in);
    cudaStatus = cudaFree(h_out);
    cudaStatus = cudaFree(worklist);
#else
    free(h_in);
    free(h_out);
    free(h_out_merge);
    cudaStatus = cudaFree(d_in);
    cudaStatus = cudaFree(d_out);
#endif
    CUDA_ERR();
    cudaDeviceSynchronize();
    timer.stop("Deallocation");
    timer.print("Deallocation", 1);

    // Release timers
    timer.release("Allocation");
    timer.release("Initialization");
    timer.release("Copy To Device");
    timer.release("Kernel");
    timer.release("Copy Back and Merge");
    timer.release("Deallocation");

    printf("Test Passed\n");
    return 0;
}
