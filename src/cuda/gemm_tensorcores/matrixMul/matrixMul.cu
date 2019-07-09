/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_fp16.h>
#include "half.hpp"
#include <math.h>


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(half2 *C, half2 *C1, half2 *A,
    half2 *B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd   = aBegin + wA - 1;



  // Step size used to iterate through the sub-matrices of A
  int aStep  = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep  = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  const float valO = 0.0f;
  volatile half2 Csub =__float2half2_rn(valO); ;
  // volatile float Csub = 0;
  // volatile double Csub1= 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ half2 As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ half2 Bs[BLOCK_SIZE][BLOCK_SIZE];



    // __shared__ double As1[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ double Bs1[BLOCK_SIZE][BLOCK_SIZE];

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    As[ty][tx] = A[a + wA * ty + tx];
    Bs[ty][tx] = B[b + wB * ty + tx];

    // As1[ty][tx] = A1[a + wA * ty + tx];
    // Bs1[ty][tx] = B1[b + wB * ty + tx];

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      
      

      Csub = fma_dmr(As[ty][k], Bs[k][tx],Csub);
      // Csub1 = fma_dmr(As[ty][k], Bs[k][tx],Csub1);
      // Csub1 = fma_dmr(__double2float_rn(As[ty][k]), __double2float_rn(Bs[k][tx]), Csub1);
      
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  C[c + wB * ty + tx] = Csub;
  // C1[c + wB * ty + tx] = Csub1;
}

// template <int BLOCK_SIZE> __global__ void MatrixMulCUDA_Half(half *C, half *C1, half *A,
//     half *B, int wA,
//     int wB) {
//   // Block index
//   int bx = blockIdx.x;
//   int by = blockIdx.y;

//   // Thread index
//   int tx = threadIdx.x;
//   int ty = threadIdx.y;

//   // Index of the first sub-matrix of A processed by the block
//   int aBegin = wA * BLOCK_SIZE * by;

//   // Index of the last sub-matrix of A processed by the block
//   int aEnd   = aBegin + wA - 1;



//   // Step size used to iterate through the sub-matrices of A
//   int aStep  = BLOCK_SIZE;

//   // Index of the first sub-matrix of B processed by the block
//   int bBegin = BLOCK_SIZE * bx;

//   // Step size used to iterate through the sub-matrices of B
//   int bStep  = BLOCK_SIZE * wB;

//   // Csub is used to store the element of the block sub-matrix
//   // that is computed by the thread
//   volatile half2 Csub = __float2half2_rn(0.0);

//  // half2 Csub1= __float2half2_rn(0.0);



//   // Loop over all the sub-matrices of A and B
//   // required to compute the block sub-matrix
//   for (int a = aBegin, b = bBegin;
//        a <= aEnd;
//        a += aStep, b += bStep) {
//     // Declaration of the shared memory array As used to
//     // store the sub-matrix of A
//     __shared__ half2 As[BLOCK_SIZE][BLOCK_SIZE];

//     // Declaration of the shared memory array Bs used to
//     // store the sub-matrix of B
//     __shared__ half2 Bs[BLOCK_SIZE][BLOCK_SIZE];

//     // Load the matrices from device memory
//     // to shared memory; each thread loads
//     // one element of each matrix
//     As[ty][tx] = __half2half2(A[a + wA * ty + tx]);
//     Bs[ty][tx] = __half2half2(B[b + wB/2 * ty + tx]);

//     // Synchronize to make sure the matrices are loaded
//     __syncthreads();

//     // Multiply the two matrices together;
//     // each thread computes one element
//     // of the block sub-matrix
// #pragma unroll

//     for (int k = 0; k < BLOCK_SIZE; ++k) {
//       Csub = __hfma2(As[ty][k], Bs[k][tx],Csub);
//      // Csub1 =__hfma2((As[ty][k]), (Bs[k][tx]),Csub1);

      
//     }

//     // Synchronize to make sure that the preceding
//     // computation is done before loading two new
//     // sub-matrices of A and B in the next iteration
//     __syncthreads();
//   }

//   // Write the block sub-matrix to device memory;
//   // each thread writes one element
//   int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
//   ((half2*)C)[c + wB/2 * ty + tx] = Csub;
//   //((half2*)C1)[c + wB/2 * ty + tx] = Csub1;
 
// }

void ConstantInit(half2 *data, int size, half2 val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}


void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void ConstantInit(double *data, int size, double val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv,
                   int block_size, const dim3 &dimsA,
                   const dim3 &dimsB) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(half2) * size_A;
  // unsigned int mem_size_A1 = sizeof(double) * size_A;
  half2 *h_A = reinterpret_cast<half2 *>(malloc(mem_size_A));
  // double *h_A1 = reinterpret_cast<double *>(malloc(mem_size_A1));

  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(half2) * size_B;
  // unsigned int mem_size_B1 = sizeof(double) * size_B;
  half2 *h_B = reinterpret_cast<half2 *>(malloc(mem_size_B));

  // double *h_B1 = reinterpret_cast<double *>(malloc(mem_size_B1));
  // Initialize host memory
  
  const float valA1 = 2.0f;
  const float valB1 = 2.0f;
  
  const half2 valA = __float2half2_rn(valA1);
  
  const half2 valB = __float2half2_rn(valB1);



  
  ConstantInit(h_A, size_A, valA);
  // ConstantInit(h_A1, size_A, valA1);
  
  ConstantInit(h_B, size_B, valB); 
  // ConstantInit(h_B1, size_B, valB1);
  


  // Allocate device memory
  half2 *d_A, *d_B, *d_C, *d_C1;
  // float *d_C1;
  // double *d_A, *d_A1,*d_B, *d_B1, *d_C, *d_C1;
  // double *d_A1, *d_B1, * d_C1;
  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(half2);
  unsigned int mem_size_C1 = dimsC.x * dimsC.y * sizeof(half2);

  half2 *h_C = reinterpret_cast<half2 *>(malloc(mem_size_C));

  half2 *h_C1 = reinterpret_cast<half2 *>(malloc(mem_size_C1));

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));

  // checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A1), mem_size_A1));

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));

  // checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B1), mem_size_B1));

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

  checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C1), mem_size_C1));

  // copy host memory to device
  checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(d_A1, h_A1, mem_size_A1, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
  // checkCudaErrors(cudaMemcpy(d_B1, h_B1, mem_size_B1, cudaMemcpyHostToDevice));

  


  // Setup execution parameters
  dim3 threads(block_size, block_size);
  dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);




  // // HALF parameters 

  // dim3 threads(block_size/2.0, block_size);
  // dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);






  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");


  MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_C1, d_A, d_B,
                                         dimsA.x, dimsB.x);
  //MatrixMulCUDA_Half<32> <<< grid, threads >>>(d_C,d_C1, d_A, d_B,
  //                                          dimsA.x, dimsB.x);

  

  printf("done\n");

  cudaDeviceSynchronize();

  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start;
  checkCudaErrors(cudaEventCreate(&start));

  cudaEvent_t stop;
  checkCudaErrors(cudaEventCreate(&stop));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, NULL));

  // Execute the kernel
  int nIter = 10;

  for (int j = 0; j < nIter; j++) {
   
      MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_C1, d_A, d_B,
                                              dimsA.x, dimsB.x);
      // MatrixMulCUDA_Half<32> <<< grid, threads >>>(d_C,d_C1, d_A, d_B,
      //                                       dimsA.x, dimsB.x);
    
  }

  // Record the stop event
  checkCudaErrors(cudaEventRecord(stop, NULL));

  // Wait for the stop event to complete
  checkCudaErrors(cudaEventSynchronize(stop));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
                     (msecPerMatrixMul / 1000.0f);
  printf(
    "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops," \
    " WorkgroupSize= %u threads/block\n",
    gigaFlops,
    msecPerMatrixMul,
    flopsPerMatrixMul,
    threads.x * threads.y);

  // Copy result from device to host
  checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_C1, d_C1, mem_size_C1, cudaMemcpyDeviceToHost));
  

  printf("Checking computed result for correctness: ");
  bool correct = true;



  // Clean up memory
  free(h_A);
  // free(h_A1);
  free(h_B);
  // free(h_B1);
  free(h_C);
  checkCudaErrors(cudaFree(d_A));
  // checkCudaErrors(cudaFree(d_A1));
  checkCudaErrors(cudaFree(d_B));
  // checkCudaErrors(cudaFree(d_B1));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(d_C1));

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}


/**
 * Program main
 */
int main(int argc, char **argv) {
  printf("[Matrix Multiply Using CUDA] - Starting...\n");

  if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
      checkCmdLineFlag(argc, (const char **)argv, "?")) {
    printf("Usage -device=n (n >= 0 for deviceID)\n");
    printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
    printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
    printf("  Note: Outer matrix dimensions of A & B matrices" \
           " must be equal.\n");

    exit(EXIT_SUCCESS);
  }

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  dim3 dimsA(8192, 8192, 1);
  dim3 dimsB(8192, 8192, 1);

  dimsA.x = 8192;
  dimsA.y = 8192;

  dimsB.x = 8192;
  dimsB.y = 8192; 


  // dim3 dimsA(4096, 4096, 1);
  // dim3 dimsB(4096, 4096, 1);

  // dimsA.x = 4096;
  // dimsA.y = 4096;

  // dimsB.x = 4096;
  // dimsB.y = 4096; 



  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y,
         dimsB.x, dimsB.y);

  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);

  exit(matrix_result);
}

__device__ __forceinline__ double fma_dmr(double a, double b, double acc) {
  return __fmaf_rn(a, b, acc);
}

__device__ __forceinline__ float fma_dmr(float a, float b, float acc) {
  return __fmaf_rn(a, b, acc);
}

__device__  __forceinline__ half2 fma_dmr(half2 a, half2 b, half2 acc) {
  return __hfma2(a, b, acc);
}
