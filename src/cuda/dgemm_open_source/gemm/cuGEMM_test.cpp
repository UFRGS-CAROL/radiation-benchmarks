/* GEMM is a General Matrix Multiply - a subroutine in the Basic Linear Algebra Subprograms library*/

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "helper_cuda.h"

/* ======================================================= */
/* CUDA implementation of dGEMM without using shared memory
/* ======================================================= */
__global__ void cuda_dgemm(int n, 
			   double alpha, 
			   const double *A, 
			   const double *B,
			   double beta, 
			   double *C) {

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  

  if (row > n || col > n) return;
  
  double prod = 0;
  int kk;
  for (kk = 0; kk < n; ++kk){
    prod += A[row * n + kk] * B[kk * n + col];
  }
  C[row*n + col] = alpha * prod + beta * C[row*n+col];
}

/* ======================================================= */
/* CUDA implementation of dGEMM using shared memory
/* ======================================================= */
__global__ void cuda_dgemm_shmem(int n, 
			   double alpha, 
			   const double *A, 
			   const double *B,
			   double beta, 
			   double *C) {
  // Block index
  int block_col = blockIdx.x;
  int block_row = blockIdx.y;

  // Thread index
  int thread_col = threadIdx.x;
  int thread_row = threadIdx.y;


  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  
  int aBegin = n * blockDim.x * block_row;
  int aEnd = aBegin + n-1;
  int bBegin = blockDim.x * block_col;
  int bStep = n * blockDim.x;
  double Csub = 0;

  for (int a=aBegin, b=bBegin, istep=0;
       a <= aEnd; a+= blockDim.x, b+=bStep, ++istep){

    __shared__ double As[blockDim.x][blockDim.x];
    __shared__ double Bs[blockDim.x][blockDim.x];

    if ((istep*blockDim.x+thread_col < n) && (block_row*blockDim.x+ thread_row < n))
      As[thread_row][thread_col] = A[a + n * thread_row + thread_col];
    else
      As[thread_row][thread_col] = 0;
      
    if ((block_col*blockDim.x+thread_col < n) && (istep*blockDim.x + thread_row < n))
      Bs[thread_row][thread_col] = B[b + n * thread_row + thread_col];
    else
      Bs[thread_row][thread_col] = 0;

    __syncthreads();

    // calculate the cell
    for (int k = 0; k < blockDim.x; ++k)
      Csub += As[thread_row][k] * Bs[k][thread_col];

    __syncthreads();
  }

  // Write the block sub-matrix to global memory;
  // each thread writes one element
  int c = n * blockDim.x * block_row + blockDim.x * block_col;
  if ((block_col*blockDim.x+thread_col < n) && (block_row*blockDim.x+ thread_row < n))
    C[c + n * thread_row + thread_col] = alpha * Csub + beta * C[c +n * thread_row + thread_col];

 }


/* ======================================================= */
/* Simple host implementation of a simple version of sgemm */
/* ======================================================= */
static void simple_dgemm(int n, double alpha, const double *A, const double *B,
                         double beta, double *C) {
  int i, j, k;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j){
      double prod = 0;
      for (k = 0; k < n; ++k){
	prod += A[k * n + i] * B[j * n + k];
      }
      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

/* ======================= */
/* dgemm from BLAS library */
/* ======================= */
extern "C"{
extern void dgemm_(char *, char * , 
		  int *, int *, int *,
		  double *, double *, int *,
		  double *, int *,
		   double *, double *, int *); };

/* ==== */
/* Main */
/* ==== */
int main(int argc, char **argv)
{
  cublasStatus_t status;
  double *h_A, *h_B, *h_C, *h_C_blas, *h_C_simple;
  double *d_A = 0; 
  double *d_B = 0;
  double *d_C = 0;
  double alpha = 1.0f;
  double beta = 0.0f;
  int n2, N;
  int i;
  double error_norm1, error_norm2;
  double ref_norm;
  double diff1, diff2;
  cublasHandle_t handle;
  struct timeval tv1, tv2;


  /* get the size of the matrix from the command line */
  if (argc <2 ) N= 275;
  else N = atoi(argv[1]);
  n2 = N * N;

  printf("\nRunning dgemm test for %d by %d matricies.\n", N, N);
  /* Initialize CUBLAS */
  status = cublasCreate(&handle);
  
  /* Allocate host memory for the matrices */
  h_A = (double *)malloc(n2 * sizeof(double) );
  h_B = (double *)malloc(n2 * sizeof(double) );
  h_C = (double *)malloc(n2 * sizeof(double) );
  h_C_blas = (double *)malloc(n2 * sizeof(double) );
  h_C_simple = (double *)malloc(n2 * sizeof(double) );

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++){
    h_A[i] = rand() / (double)RAND_MAX;
    h_B[i] = rand() / (double)RAND_MAX;
    h_C[i] = rand() / (double)RAND_MAX;
    h_C_blas[i] = h_C[i];
    h_C_simple[i] = h_C[i];
  }

  printf("\tTesting simple C implementation of dgemm function.\n");
  gettimeofday(&tv1, NULL);
  /* Performs operation using plain C code */
  simple_dgemm(N, alpha, h_A, h_B, beta, h_C_simple);
  gettimeofday(&tv2, NULL);
  printf("\t\tdone...\n");
  printf("\t\tExecution time (in millisec): %.2f\n",
	 (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
	 (double)(tv2.tv_sec -tv1.tv_sec )*1000);


  printf("\tTesting dgemm function from BLAS library.\n");
  gettimeofday(&tv1, NULL);

  /* Performs operation using BLASS library */
  dgemm_("N","N", &N, &N, &N, &alpha, h_A, &N, h_B, &N, &beta, h_C_blas, &N);
  gettimeofday(&tv2, NULL);
  printf("\t\tdone...\n");
  printf("\t\tExecution time (in millisec): %.2f\n",
	 (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
	 (double)(tv2.tv_sec -tv1.tv_sec )*1000);


  printf("\tTesting dgemm function from cuBLAS library.\n");
  gettimeofday(&tv1, NULL);

  /* Allocate device memory for the matrices */
  cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0]));
  cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0]));
  cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0]));

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

  /* Performs operation using cublas */
  status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

  /* Read the result back */
  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

  gettimeofday(&tv2, NULL);
  printf("\t\tdone...\n");
  printf("\t\tExecution time (in millisec): %.2f\n",
	 (double)(tv2.tv_usec-tv1.tv_usec)/1000 + 
	 (double)(tv2.tv_sec -tv1.tv_sec )*1000);

  printf("\n\tChecking results.\n");
  /* Check result against reference */
  error_norm1 = 0;
  error_norm2 = 0;
  ref_norm = 0;
  for (i = 0; i < n2; ++i){
    diff1 = h_C_simple[i] - h_C[i];
    diff2 = h_C_simple[i] - h_C_blas[i];
    error_norm1 += diff1 * diff1;
    error_norm2 += diff2 * diff2;
    ref_norm += h_C_simple[i] * h_C_simple[i];
  }

  error_norm1 = (double)sqrt((double)error_norm1);
  error_norm2 = (double)sqrt((double)error_norm2);
  ref_norm = (double)sqrt((double)ref_norm);

  if (fabs(ref_norm) < 1e-7)printf(" *** Error in Calculations! \n");
  if(error_norm1 / ref_norm < 1e-6f)printf("\t\tPassed cublas Dgemm vs. simple Dgemm comparison!\n");
  else printf("\t\tDid not pass cublas Dgemm vs. simple Dgemm comparison!\n");
  if(error_norm2 / ref_norm < 1e-6f)printf("\t\tPassed simple Dgemm vs. BLAS Dgemm comparison!\n");
  else printf("\t\tDid not pass simple Dgemm vs. BLAS Dgemm comparison!\n");

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_simple);
  free(h_C_blas);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


  /* Shutdown */
  status = cublasDestroy(handle);

  return(0);
}
