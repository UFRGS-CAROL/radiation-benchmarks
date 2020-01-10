#ifndef NEEDLE_H_
#define NEEDLE_H_

#ifdef RD_WG_SIZE_0_0

#define BLOCK_SIZE RD_WG_SIZE_0_0

#elif defined(RD_WG_SIZE_0)

#define BLOCK_SIZE RD_WG_SIZE_0

#elif defined(RD_WG_SIZE)

#define BLOCK_SIZE RD_WG_SIZE

#else

#define BLOCK_SIZE 32 //16

#endif

typedef unsigned KErrorsType;

__device__ KErrorsType gpukerrors;

__global__ void needle_cuda_shared_2(int* referrence, int* matrix_cuda,
		int cols, int penalty, int i, int block_width);
__global__ void needle_cuda_shared_1(int* referrence, int* matrix_cuda,
		int cols, int penalty, int i, int block_width);
__global__ void GoldChkKernel(int *gk, int *ck, int n);

#define GCHK_BLOCK_SIZE 32

#endif
