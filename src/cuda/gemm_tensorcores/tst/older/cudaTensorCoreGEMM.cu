#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

// Externally configurable parameters.
//
#ifndef SHARED_MEMORY_LIMIT_64K
// Set this to 0 to use more than 64 Kb of shared memory to cache data, to
// improve the performance of the computations on GPU.
// Note that you need a GPU that can have more than 64 Kb of shared memory
// per multiprocessor.
#define SHARED_MEMORY_LIMIT_64K 1
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

#define BLOCK_SIZE 32

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

//using namespace nvcuda;

__host__ void init_host_matrices(float *a, float *b, float *c) {
	for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
		a[t] = 1.0;
		b[t] = 1.0;
		c[t] = 1.0;
	}
}

__host__ void init_host_matrices(half *a, half *b, half *c) {
	for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
		a[t] = 1.0;
		b[t] = 1.0;
		c[t] = 1.0;
	}
}

template<typename T>
__global__ void MatrixMulCUDA(const T *A, const T *B, const T *C, T* D,
		const T alpha, const T beta, const int wA, const int wB) {
	register int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	register int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;

	register T acc = 0.0;
	for (int k = 0; k < wB; k++) {
		acc += A[ty * wA + k] * B[k * wB + tx];
	}

	D[ty * wA + tx] = acc * alpha + beta * C[ty * wA + tx];

}

__global__ void MatrixMulCUDA(const float *A, const float *B, const float *C,
		float* D, const float alpha, const float beta, const int wA,
		const int wB) {
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
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

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
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;

	D[c + wB * ty + tx] = alpha * Csub + beta * C[c + wB * ty + tx];
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

int main(int argc, char **argv) {
	printf("Initializing...\n");

	int dev = findCudaDevice(argc, (const char **) argv);

	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

	// Tensor cores require a GPU of Volta (SM7X) architecture or higher.
	if (deviceProp.major < 7) {
		printf("cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
				"Cores.  Exiting...\n");
		exit (EXIT_WAIVED);
	}

	printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
	printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
	printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

	half *A_h = NULL;
	half *B_h = NULL;
	half *C_h = NULL;
	half *D_h = NULL;

	A_h = (half *) malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
	B_h = (half *) malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
	C_h = (half *) malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
	D_h = (half *) malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);
	half *A = NULL;
	half *B = NULL;
	half *C = NULL;
	half *D = NULL;

	checkCudaErrors(
			cudaMalloc((void**) &A, sizeof(half) * M_GLOBAL * K_GLOBAL));
	checkCudaErrors(
			cudaMalloc((void**) &B, sizeof(half) * N_GLOBAL * K_GLOBAL));
	checkCudaErrors(
			cudaMalloc((void**) &C, sizeof(half) * M_GLOBAL * N_GLOBAL));
	checkCudaErrors(
			cudaMalloc((void**) &D, sizeof(half) * M_GLOBAL * N_GLOBAL));

	assert(((unsigned long long) A) % 128 == 0);
	assert(((unsigned long long) B) % 128 == 0);
	assert(((unsigned long long) C) % 128 == 0);
	assert(((unsigned long long) D) % 128 == 0);

	half* dtd = nullptr;
	half* dt = (half *) malloc(sizeof(half) * M_GLOBAL * N_GLOBAL);

	checkCudaErrors(
			cudaMalloc((void**) &dtd, sizeof(half) * M_GLOBAL * N_GLOBAL));

	//INIT HOST
	init_host_matrices(A_h, B_h, C_h);

	checkCudaErrors(cudaMemset(dtd, 0, sizeof(half) * M_GLOBAL * N_GLOBAL));

	printf("Preparing data for GPU...\n");

	checkCudaErrors(
			cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL,
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL,
					cudaMemcpyHostToDevice));
	checkCudaErrors(
			cudaMemcpy(C, C_h, sizeof(half) * M_GLOBAL * N_GLOBAL,
					cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemset(D, 0, sizeof(half) * M_GLOBAL * N_GLOBAL));

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

	printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

	const half alpha = 1.0;
	const half beta = 1.0;
	cudaEvent_t start, stop;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start));

	// If enough shared memory available on the GPU use high performant kernel

	printf("Computing... using high performance kernel compute_gemm \n");

	cudaStream_t st;
	cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
	std::cout << BLOCK_SIZE << " " << M_GLOBAL << std::endl;
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid( M_GLOBAL / threads.x, M_GLOBAL / threads.y);

	checkCudaErrors(
			cudaFuncSetAttribute(compute_gemm,
					cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	checkCudaErrors(
			cudaFuncSetAttribute(MatrixMulCUDA<half>,
					cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

	compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ,
			st>>>(A, B, C, dtd, alpha, beta, M_GLOBAL,
	M_GLOBAL);

	MatrixMulCUDA<half> <<<grid, threads, SHMEM_SZ>>>(A, B, C, D, alpha, beta,
	M_GLOBAL, M_GLOBAL);

	checkKernelErrors(cudaStreamSynchronize(st));
	checkKernelErrors(cudaPeekAtLastError());
	checkKernelErrors(cudaDeviceSynchronize());

	checkCudaErrors(
			cudaMemcpy(D_h, D, sizeof(half) * M_GLOBAL * N_GLOBAL,
					cudaMemcpyDeviceToHost));
	checkCudaErrors(
			cudaMemcpy(dt, dtd, sizeof(half) * M_GLOBAL * N_GLOBAL,
					cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaEventSynchronize(stop));

	printf("Verifying correctness of the computations...\n");

	// memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

	// matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
	//                   K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

	for (int i = 0; i < 10; i++) {
		printf(" diff = %f, HW = %f, SW = %f \n",
				(double(D_h[i]) - double(dt[i])), double(dt[i]),
				double(D_h[i]));
	}

	float milliseconds = 0;

	checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

	printf("Time: %f ms\n", milliseconds);
	printf("TFLOPS: %.2f\n",
			static_cast<double>((static_cast<double>(M_GLOBAL) *
			N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);

	free(A_h);
	free(B_h);
	free(C_h);
	free(D_h);
	free(dt);
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(dtd)));

	checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
	checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

	cudaStreamDestroy(st);

	return 0;
}
