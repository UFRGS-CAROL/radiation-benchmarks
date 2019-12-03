/* Copyright (c) 1993-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include <stdint.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}


#include <mma.h>
using namespace nvcuda;

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

#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT nvcuda::wmma::mem_row_major

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


// Must be multiples of 16 for wmma code to work
#define MATRIX_M 4096 //16384
#define MATRIX_N 4096 //16384
#define MATRIX_K 4096 //16384

#define BLOCK_SIZE 32




// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__device__ __forceinline__ void axpy__(const double a, const double b, double &c) {
    c = __fma_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, float &c) {
    //printf("A = %f   -- B =  %f\n", a, b);
    c = __fmaf_rn(a, b, c);
}
__device__ __forceinline__ void axpy__(const double a, const double b, float &c) {
    c = __fmaf_rn(__double2float_rn(a), __double2float_rn(b), c);
}
__device__ __forceinline__ void axpy__(const float a, const float b, __half &c) {
    c = __hfma(__float2half(a), __float2half(b), c);
}

__device__  __forceinline__ half axpy__(half a, half b, half acc) {
  return __hfma(a, b, acc);
}



__global__ void compute_gemm(const half *A, const half *B, const half *C,
    half *D, half alpha, half beta, int wA, int wB) {
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
        // Copy 16 bytes at once in each lane.
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

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.

__global__ void wmma_example(half *a, half *b, half *c, int M, int N, int K, half alpha, half beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
 
         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.


// __global__ void wmma_example_dmr(half *a, half *b, float *c, float *d_sw, float *d_wmma, int M, int N, int K, float alpha, float beta) {

//   // Leading dimensions. Packed with no transpositions.
//   int lda = M;
//   int ldb = K;
//   int ldc = M;
 

//   // Tile using a 2D grid
//   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

//   // Declare the fragments
//   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
//   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
//   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
//   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
      
//   int row = blockIdx.x * blockDim.x + threadIdx.x;
//   int col = blockIdx.y * blockDim.y + threadIdx.y;

        
//     if (row < M && col < N) {
//       register float acc_real_t = 0.0;
         

     
//       for (int i = 0; i < K; i++) {
//         axpy__((float)a[row * M + i], (float)b[col * N + i], acc_real_t);
//       }   
       

     

//       d_sw[row * M + col] = acc_real_t;
        
//     }
    
    
    
    
//   wmma::fill_fragment(acc_frag, 0.0f);

//     // Loop over k
//   for (int i = 0; i < K; i += WMMA_K) {
//     int aRow = warpM * WMMA_M;
//     int aCol = i;

//     int bRow = i;
//     int bCol = warpN * WMMA_N;

    
//     // Bounds checking
//     if (aRow < M && aCol < K && bRow < K && bCol < N) {
//          // Load the inputs
//       wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
//       wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

//          // Perform the matrix multiplication
//       wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

//     }
// }

//    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
// int cRow = warpM * WMMA_M;
// int cCol = warpN * WMMA_N;

// if (cRow < M && cCol < N) {
//   wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


//   for(int i=0; i < c_frag.num_elements; i++) {
   
//     c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
//     if (row < M && col < N) {
      
//       register float acc_real_t = 0.0;

//       //for (int internal = i; internal < WMMA_N; internal++) {
//       //  axpy__((float)a[row * M + internal], (float)b[col * N + internal], acc_real_t);    
//       for (int i = 0; i < K; i++) {
//         acc_real_t += (float)a[row * M + i] * (float)b[col * N + i];
//       }   
      
//       d_sw[row * M + col] = acc_real_t * alpha + beta * c[row * M + col];
//     }      
//   }

//       // Store the output
//   wmma::store_matrix_sync(d_wmma + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
// }
   

// }



__global__ void matrix_mult(half *A, half *B, int wA,
    int wB, half *C, half alpha, half beta) {
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
  half Csub = 0;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) {
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

    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Csub += As[ty][k] * Bs[k][tx];
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  //int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
  const int index = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx + wB * ty + tx;
  half half_val = alpha * Csub + beta * C[index];
  C[index] = Csub;
}



__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}



int main(int argc, char* argv[]) {
  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  half *c;
  half *c_wmma;
  half *c_cublas;
  half *d_wmma;
  half *d_sw;  
  half *d_host_cublas;
  half *d_host_wmma;
  half *d_host_sw;
  

  
  curandGenerator_t gen;
  cublasHandle_t cublasHandle;
  
  cudaEvent_t startWMMA;
  cudaEvent_t stopWMMA;
  cudaEvent_t startMXM;
  cudaEvent_t stopMXM;

  cudaEvent_t startcublas;
  cudaEvent_t stopcublas;
   
  cudaErrCheck(cudaEventCreate(&startWMMA));
  cudaErrCheck(cudaEventCreate(&stopWMMA));

  cudaErrCheck(cudaEventCreate(&startMXM));
  cudaErrCheck(cudaEventCreate(&stopMXM));
  
  cudaErrCheck(cudaEventCreate(&startcublas));
  cudaErrCheck(cudaEventCreate(&stopcublas));
   
   
  cublasErrCheck(cublasCreate(&cublasHandle));
   
  // Use tensor cores
  cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
  
  cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&d_sw, MATRIX_K * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)&d_wmma, MATRIX_K * MATRIX_N * sizeof(half)));

  d_host_cublas = (half*)malloc(MATRIX_M * MATRIX_N * sizeof(half));
  d_host_wmma = (half*)malloc(MATRIX_M * MATRIX_N * sizeof(half));
  d_host_sw = (half*)malloc(MATRIX_M * MATRIX_N * sizeof(half));

  
   
   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

  
   
   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   //curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));
   
   curandErrCheck(curandDestroyGenerator(gen));
   


  //cudaErrCheck(cudaMemset(a_fp16, 6462.8195679, MATRIX_M * MATRIX_N * sizeof(half)));
  //cudaErrCheck(cudaMemset(b_fp16, 6462.8195679, MATRIX_M * MATRIX_N * sizeof(half)));

  cudaErrCheck(cudaMemset(c_cublas, 0.0f, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMemset(c_wmma, 0.0f, MATRIX_M * MATRIX_N * sizeof(half)));
  
  cudaErrCheck(cudaMemset(d_sw, 0.0f, MATRIX_M * MATRIX_N * sizeof(half)));
  cudaErrCheck(cudaMemset(d_wmma, 0.0f, MATRIX_M * MATRIX_N * sizeof(half)));



 
  half alpha = 1.0f;
  half beta = 1.0f;


   
   

  // WMMA TENSOR //
  dim3 gridDim;
  dim3 blockDim;

  dim3 dim_grid, dim_block;
  
  uint32_t grid_rows = (MATRIX_M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  uint32_t grid_cols = (MATRIX_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim_grid = dim3(grid_cols, grid_rows);
  dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
 
  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   
  //printf("Running with wmma thread dimensions...\n");
  //cudaErrCheck(cudaEventRecord(startWMMA));
  //wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  
  //cudaErrCheck(cudaEventRecord(stopWMMA));

   

   // MXM DIMENSIONS
   
  //blockDim.x = WMMA_M; //128;
  //blockDim.y = WMMA_N;
  //blockDim.x = 128;
  //blockDim.y = 4;

  //printf("Running  mxm with MXM thread dimensions...\n");

  enum {
    // Compute the right amount of shared memory to request.
    // We need shared memory to hold per-CTA C and D matrix tiles, and to cache
    // per-CTA chunks
    // of the A and B matrices. Therefore, the right amount to request is the
    // maximum of those
    // two numbers.
    SHMEM_SZ = MAX(
        sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF)
            * 2,
        M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N
            * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(half))
  };

 
  cudaStream_t stream1, stream2;
  cudaErrCheck(cudaStreamCreate(&stream1)); 
  cudaErrCheck(cudaStreamCreate(&stream2));
  int ite = 10; 
  //printf("Running  dmr with MXM thread dimensions...\n");
  cudaErrCheck(cudaEventRecord(startMXM));


   for (int i = 0; i < ite; ++i)
   {
     // ---- MXM SW ----//
    matrix_mult<<< dim_grid, dim_block,0, stream1 >>> (a_fp16, b_fp16, MATRIX_M, MATRIX_N, d_sw, alpha, beta);

    //wmma_example <<< gridDim, blockDim,0, stream2 >>> (a_fp16, b_fp16, d_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
    compute_gemm <<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ,
      stream2>>> (A, B, C, d_wmma, alpha, beta, M_GLOBAL,  M_GLOBAL);

    cudaErrCheck(cudaDeviceSynchronize());
   }


   
   // ---- DMR --- //
  //printf("Running  dmr with tensor thread dimensions...\n");
  
  //wmma_example_dmr <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, d_sw, d_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
  cudaErrCheck(cudaEventRecord(stopMXM));
 
   


  /* 
  // Now using cuBLAS
  printf("Running with cuBLAS...\n");
  cudaErrCheck(cudaEventRecord(startcublas));
  cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta, 
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  cudaErrCheck(cudaEventRecord(stopcublas));
  */ 


  // Error checking
  printf("\nChecking results...\n");
  //cudaErrCheck(cudaMemcpy(d_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(d_host_sw, d_sw, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(d_host_wmma, d_wmma, MATRIX_M * MATRIX_N * sizeof(half), cudaMemcpyDeviceToHost));



    for (int i = 0; i <  20; i++) {      
    float v1 = d_host_wmma[i];
    float v2 = d_host_sw[i];
    //float v3 = d_host_cublas[i]; 
    //half v4 = v2/v1;     
    printf("TENSOR = %f  | ------  MXM = %f  ----- | \n", v1, v2);

  }
   
  
  //float wmmaTime;
  //float cublasTime;
  float mxmTime;
  //cudaErrCheck(cudaEventSynchronize(stopWMMA));
  //cudaErrCheck(cudaEventSynchronize(stopcublas));
  cudaErrCheck(cudaEventSynchronize(stopMXM));

  //cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
  //cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
  cudaErrCheck(cudaEventElapsedTime(&mxmTime, startMXM, stopMXM));
  //printf("wmma took %fms\n", wmmaTime);
  //printf("cublas took %fms\n", cublasTime);
  printf("mxm took %fms\n", mxmTime);
  
     
   
 
 
 cudaErrCheck(cudaEventDestroy(startWMMA));
 cudaErrCheck(cudaEventDestroy(stopWMMA));

 cudaErrCheck(cudaEventDestroy(startcublas));             
 cudaErrCheck(cudaEventDestroy(stopcublas));
 
 cudaErrCheck(cudaFree(a_fp32));
 cudaErrCheck(cudaFree(b_fp32));
 cudaErrCheck(cudaFree(a_fp16));
 cudaErrCheck(cudaFree(b_fp16));
 cudaErrCheck(cudaFree(d_wmma));
 cudaErrCheck(cudaFree(d_sw));

 cudaErrCheck(cudaFree(c));
 cudaErrCheck(cudaFree(c_cublas));
 cudaErrCheck(cudaFree(c_wmma));

 free(d_host_cublas);
 free(d_host_wmma);
 free(d_host_sw);

 cudaErrCheck(cudaDeviceReset());
 return 0;
}
