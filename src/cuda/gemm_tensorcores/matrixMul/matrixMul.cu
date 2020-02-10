
#include "/home/carol/radiation-benchmarks/src/cuda/common/include/device_vector.h"
#include <vector>
#include <iostream>

#include <cuda_fp16.h>

#define BLOCK_SIZE 32


__global__ void matrix_mult_kernel_unhardened(  //Kernel without hardening
        half *A,  //A
        half *B,  //B
        half *C,  //C
        half alpha, half beta, int wA, int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    half Csub = 0;
    half2 Csub_h2(0, 0);
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ half As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE / 2; ++k) {
            auto a = __halves2half2(As[ty][k], As[ty][k + 1]);
            auto b = __halves2half2(Bs[k][tx], Bs[k + 1][tx]);
            Csub_h2 = __hfma2(a, b, Csub_h2);
            //Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    const int index = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;
    C[index] = alpha * (Csub_h2.x + Csub_h2.y) + beta * C[index];
}

int main(){
    constexpr auto n = 1 << 12;
    constexpr auto size = n * n;
    std::cout << "Size " << n << " elements " << size << std::endl;
    std::vector<half> ah(size, 1.0), bh(size, 1.0), ch(size, 0);
    rad::DeviceVector<half> ad = ah;
    rad::DeviceVector<half> bd = bh;
    rad::DeviceVector<half> cd = ch;
    
    uint32_t grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto dim_grid = dim3(grid_cols, grid_rows);
    auto dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_kernel_unhardened<<<dim_grid, dim_block>>>(ad.data(), bd.data(), cd.data(), half(1.0), half(0.0), n, n);
    rad::checkFrameworkErrors(cudaDeviceSynchronize());
    rad::checkFrameworkErrors(cudaPeekAtLastError());
    
    cd.to_vector(ch);
    for(auto i : ch){
        if(float(i) != float(n))
            throw "Bad result\n";
    }
    std::cout << "Good result\n";
}
