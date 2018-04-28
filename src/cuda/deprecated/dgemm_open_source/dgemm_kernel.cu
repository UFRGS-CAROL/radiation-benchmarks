#define BLOCK_SIZE_KERNEL 16
#define BLOCK_SIZE 16

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ int kerrors;

__global__ void GoldChkKernel(double *gk, double *ck, int n) //, int *kerrors)
		{
//================== HW Accelerated output validation
	int tx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if ((fabs((gk[ty * n + tx] - ck[ty * n + tx]) / gk[ty * n + tx])
			> 0.0000000001)
			|| (fabs((gk[ty * n + tx] - ck[ty * n + tx]) / ck[ty * n + tx])
					> 0.0000000001))
		atomicAdd(&kerrors, 1);

}

/* ======================================================= */
/* CUDA implementation of dGEMM using shared memory
/* ======================================================= */
__global__ void cuda_dgemm_shmem(int n,
			   double alpha,
			   const double *B,
			   const double *A,
			   double beta,
			   double *C) {
  // Block index
  int block_col = blockIdx.x;
  int block_row = blockIdx.y;

  // Thread index
  int thread_col = threadIdx.x;
  int thread_row = threadIdx.y;

  //printf("row = %d col = %d  n= %d\n", block_col, block_row, n);
  //int row = blockDim.y * blockIdx.y + threadIdx.y;
  //int col = blockDim.x * blockIdx.x + threadIdx.x;

  int aBegin = n * blockDim.x * block_row;
  int aEnd = aBegin + n-1;
  int bBegin = blockDim.x * block_col;
  int bStep = n * blockDim.x;
  double Csub = 0;

  for (int a=aBegin, b=bBegin, istep=0;
       a <= aEnd; a+= blockDim.x, b+=bStep, ++istep){

    __shared__ double As[BLOCK_SIZE_KERNEL][BLOCK_SIZE_KERNEL];
    __shared__ double Bs[BLOCK_SIZE_KERNEL][BLOCK_SIZE_KERNEL];

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

