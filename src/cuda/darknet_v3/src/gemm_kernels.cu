/**
 * GEMM kernels open source
 */
#include "gemm_kernels.h"
#include "stdio.h"

#include "cuda_fp16.h"

#define BLOCK_SIZE 32

//extern "C" void check_error(cudaError_t status);

extern "C" {
#include "cuda.h"
}

template<class tested_type>
__global__ void MatrixMulKernel(tested_type *a, tested_type *b, tested_type *c,
		int N, int M, int K, int lda, int ldb, int ldc) {

	//N
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	//M
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (M < ty || N < tx) {
		return;
	}
	tested_type acc = 0.0;
	for (int k = 0; k < K; k++) {
		acc = a[ty * lda + k] * b[k * ldb + tx] + acc;
	}

	c[ty * ldc + tx] = acc;
}

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
//#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A,
		int lda, float const *B, int ldb, float beta, float *C, int ldc) {
	/*
	 // Define type definition for single-precision CUTLASS GEMM with column-major
	 // input matrices and 128x128x8 threadblock tile size (chosen by default).
	 //
	 // To keep the interface manageable, several helpers are defined for plausible compositions
	 // including the following example for single-precision GEMM. Typical values are used as
	 // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
	 //
	 // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

	 using ColumnMajor = cutlass::layout::ColumnMajor;

	 using CutlassGemm = cutlass::gemm::device::Gemm<float, // Data-type of A matrix
	 ColumnMajor,// Layout of A matrix
	 float,// Data-type of B matrix
	 ColumnMajor,// Layout of B matrix
	 float,// Data-type of C matrix
	 ColumnMajor>;
	 // Layout of C matrix

	 // Define a CUTLASS GEMM type
	 CutlassGemm gemm_operator;

	 // Construct the CUTLASS GEMM arguments object.
	 //
	 // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
	 // in host code and passed to kernels by value. These may include pointers, strides, scalars,
	 // and other arguments needed by Gemm and its components.
	 //
	 // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
	 // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
	 //
	 CutlassGemm::Arguments args( { M, N, K },  // Gemm Problem dimensions
	 { A, lda },    // Tensor-ref for source matrix A
	 { B, ldb },    // Tensor-ref for source matrix B
	 { C, ldc },    // Tensor-ref for source matrix C
	 { C, ldc }, // Tensor-ref for destination matrix D (may be different memory than source C matrix)
	 { alpha, beta }); // Scalars used in the Epilogue

	 //
	 // Launch the CUTLASS GEMM kernel.
	 //

	 cutlass::Status status = gemm_operator(args);

	 //
	 // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
	 //

	 if (status != cutlass::Status::kSuccess) {
	 return cudaErrorUnknown;
	 }

	 // Return success, if no errors were encountered.
	 * */
	return cudaSuccess;

}

#if __CUDA_ARCH__ > 600
void hgemm(int b_operation, int a_operation, int N, int M, int K,
		half *alpha, half* b_gpu, int ldb, half* a_gpu, int lda, half* beta,
		half* c_gpu, int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

//	printf("M %d N %d K %d lda %d, ldb %d, ldc %d, x %d y %d\n", M, N, K, lda, ldb, ldc, grid.x, grid.y);
	MatrixMulKernel<<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb, ldc);
	check_error(cudaError_t(cudaPeekAtLastError()));
}
#endif

void sgemm(int b_operation, int a_operation, int N, int M, int K, float *alpha,
		float* b_gpu, int ldb, float* a_gpu, int lda, float* beta, float* c_gpu,
		int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

	MatrixMulKernel<<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb,
			ldc);
//	cudaError_t error = CutlassSgemmNN(M, N, K, *alpha, b_gpu, lda, a_gpu, ldb,
//			*beta, c_gpu, ldc);
	check_error(error);
}

void dgemm(int b_operation, int a_operation, int N, int M, int K, double *alpha,
		double* b_gpu, int ldb, double* a_gpu, int lda, double* beta,
		double* c_gpu, int ldc) {
	int gridsize_n = ceil(N / BLOCK_SIZE);
	int gridsize_m = ceil(M / BLOCK_SIZE);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(gridsize_n, gridsize_m);

	MatrixMulKernel<<<grid, threads>>>(a_gpu, b_gpu, c_gpu, N, M, K, lda, ldb,
			ldc);
	check_error(cudaError_t(cudaPeekAtLastError()));
}
