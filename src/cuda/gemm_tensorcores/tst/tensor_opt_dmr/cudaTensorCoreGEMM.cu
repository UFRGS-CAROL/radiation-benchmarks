#include "/home/carol/radiation-benchmarks/src/cuda/common/include/device_vector.h"
#include <vector>
#include <iostream>
#include <cuda_fp16.h>
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <random>
// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 0
#endif

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.

#define M_TILES 256 //512 // 128 for 2k, 512 for 8k etc 
#define N_TILES 256 //512 //
#define K_TILES 256 //512 //


#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define BLOCK_SIZE 32

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4

#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)



// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

using namespace nvcuda;

__global__ void compute_gemm(const half *A, const half *B, const half *C,
        half *D, half alpha, half beta) {
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    half *shmem_warp_tile_ptr = (half *) &shmem[0][0]
            + (warpId / 2) * SHMEM_STRIDE * K * 2+
            (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    half *shmem_warp_stream_ptr = (half *) &shmem[0][0]
            + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES)
                / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES)
                % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES) {
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE
                + block_tile_j * N;
        const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
#pragma unroll
        for (int i = 0; i < K; i++) {
            typedef int4 copy_t;

            *((copy_t *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                    *((copy_t *) (src_gmem_warp_stream_ptr
                            + GLOBAL_MEM_STRIDE * i) + laneId);
        }

        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        nvcuda::wmma::fragment < nvcuda::wmma::accumulator, M, N, K, half
                > c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                const half *tile_ptr = shmem_warp_tile_ptr
                        + i * SHMEM_STRIDE * K + j * N;

                nvcuda::wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE,
                C_LAYOUT);
            }
        }

        __syncthreads();

        // Scale the C matrix.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) {
                    c[i][j].x[t] *= beta;
                }
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr =
                (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                M * K_GLOBAL * (warpId % 4) * 2) :
                                (&B[block_tile_j * N * K_GLOBAL] +
                                N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
#pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx =
                    warpId < (WARPS_PER_BLOCK / 2) ?
                            (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2) :
                            (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2
                                    + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *) (warp_ptr + tile_k * K
                    + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL)
                    + (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

#pragma unroll
            for (int i = 0;
                    i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2;
                    i++) {
                // Copy 16 bytes dodasat once in each lane.
                *((int4 *) &shmem[shmem_idx][0]
                        + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *) ((half *) lane_ptr
                        + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
#pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                nvcuda::wmma::fragment < nvcuda::wmma::matrix_a, M, N, K, half, nvcuda::wmma::row_major
                        > a[WARP_COL_TILES];
                nvcuda::wmma::fragment < nvcuda::wmma::matrix_b, M, N, K, half, nvcuda::wmma::col_major
                        > b[WARP_ROW_TILES];

#pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    nvcuda::wmma::load_matrix_sync(a[i], tile_ptr,
                    K * CHUNK_K + SKEW_HALF);

#pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off
                                    + (WARP_ROW_TILES * N) * (warpId % 2)
                                    + (j * N);
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step
                                    * K];

                            nvcuda::wmma::load_matrix_sync(b[j], tile_ptr,
                            K * CHUNK_K + SKEW_HALF);
                        }

                        nvcuda::wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }

            __syncthreads();
        }

        // Store the D fragments to shared memory.
#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K
                        + j * N;

                nvcuda::wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE,
                C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        half *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
        for (int i = 0; i < K; i++) {
            *((int4 *) (dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i)
                    + laneId) = *((int4 *) (shmem_warp_stream_ptr
                    + SHMEM_STRIDE * i) + laneId);
        }

        __syncthreads();
    }
}




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

int main(int argc, char **argv){
    constexpr auto n = M_GLOBAL;
    constexpr auto size = n * n;
    std::cout << "Size " << n << " elements " << size << std::endl;
    // host matrices
    //std::vector<half> a(size, 1.0), b(size, 1.0), c(size, 0), d(size, 0);


    // get a number in the range 0.1 - 5.0
    std::random_device rd; //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 5.0);

    half input  = (half)dis(gen); 
    
   
    std::cout << "input value = " << (float)input << std::endl;
    std::vector<half> a(size, input), b(size, input), c(size, 0), d(size, 0);    


    //device matrices  - a,b,c duplicated 
    rad::DeviceVector<half> a_s = a;
    rad::DeviceVector<half> b_s = b;
    rad::DeviceVector<half> c_s = c;

    rad::DeviceVector<half> a_h = a;
    rad::DeviceVector<half> b_h = b;
    rad::DeviceVector<half> c_h = c;
    rad::DeviceVector<half> d_h = d;

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

   	 
    cudaStream_t stream1, stream2;
  	checkKernelErrors(cudaStreamCreate(&stream1)); 
  	checkKernelErrors(cudaStreamCreate(&stream2));

  	int dev = findCudaDevice(argc, (const char **) argv);
  	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    



   


    int count = 100;
    for (int i = 0; i < count; i++)
    {
    
    //TENSOR CORES PARAMETERS
    enum {
    //  // Compute the right amount of shared memory to request.
    // // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // // per-CTA chunks
    // // of the A and B matrices. Therefore, the right amount to request is the
    // // maximum of those
    // // two numbers.
    SHMEM_SZ = MAX(
        sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
        M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
           (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(half))
    }; 
    checkCudaErrors(cudaFuncSetAttribute(
        compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

    //checkKernelErrors(
    //    (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
    //                   SHMEM_SZ, stream1>>>(a_h.data(), b_h.data(), c_h.data(), d_h.data(), half(1.0), half(1.0))));

    

     // SW MXM PARAMETERS
    uint32_t grid_rows = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    auto dim_grid = dim3(grid_cols, grid_rows);
    auto dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult_kernel_unhardened<<<dim_grid, dim_block,0,stream2>>>(a_s.data(), b_s.data(), c_s.data(), half(1.0), half(1.0), n, n);
    
    
    rad::checkFrameworkErrors(cudaDeviceSynchronize());
    rad::checkFrameworkErrors(cudaPeekAtLastError());
    }
  



    
    c_s.to_vector(c);
    d_h.to_vector(d);

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);



    //print first 5 values of each execution 
    for (int i = 0; i < 5; ++i)
    {

    	printf("sw  == %f || hw == %f  || diff = %f \n", float(c[i]), float(d[i]), (float(d[i])- float(c[i])));


    }

    
        
  
}
