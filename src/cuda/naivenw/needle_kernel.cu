__device__ __host__ int 
maximum( int a,
		 int b,
		 int c){

int k;
if( a <= b )
k = b;
else 
k = a;

if( k <=c )
return(c);
else
return(k);

}

__device__ int gpukerrors;
__global__ void GoldChkKernel(int *gk, int *ck, int n)
{
	int tx = blockIdx.x * GCHK_BLOCK_SIZE + threadIdx.x;
	int ty = blockIdx.y * GCHK_BLOCK_SIZE + threadIdx.y;
	if (gk[ty*n + tx]!=ck[ty*n + tx])
		atomicAdd(&gpukerrors, 1);

}

__global__ void
needle_cuda_multiblock_1(
			  int* referrence,
			  int* matrix_cuda, 
			  int* matrix_cuda_out, 
			  int max_cols,
			  int penalty,
			  int i) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int idx = bx * blockDim.x + tx;
   
	  if ( idx <= i ){

		  int index = (idx + 1) * max_cols + (i + 1 - idx);
	      matrix_cuda[index]= maximum( matrix_cuda[index-1-max_cols]+ referrence[index], 
			                           matrix_cuda[index-1]         - penalty, 
									   matrix_cuda[index-max_cols]  - penalty);

	  }

}

__global__ void
needle_cuda_multiblock_2(
			  int* referrence,
			  int* matrix_cuda, 
			  int* matrix_cuda_out, 
			  int max_cols,
			  int penalty,
			  int i //iter
			  ) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int idx = bx * blockDim.x + tx;
   
	  if ( idx <= i ){

		  int index =  ( max_cols - idx - 2 ) * max_cols + idx + max_cols - i - 2 ;
   		  matrix_cuda[index]= maximum( matrix_cuda[index-1-max_cols]+ referrence[index], 
			                           matrix_cuda[index-1]         - penalty, 
									   matrix_cuda[index-max_cols]  - penalty);

	  }

}





