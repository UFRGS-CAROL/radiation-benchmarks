#include <stdio.h>

#define THREAD_SIZE 8

__global__ void sortVerifyKernel(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *errNum)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint iterator;

	#pragma unroll
	for (iterator = 0; iterator < THREAD_SIZE; iterator++)
		if ((d_SrcKey[d_DstVal[idx + iterator]] != d_DstKey[idx + iterator]))
			atomicAdd(errNum, 1);
}

extern "C" uint sortVerify(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, int size)
{
	uint *d_errNum, h_errNum = 0, blockSize, numBlocks;
	
	cudaMalloc((void **)&d_errNum, sizeof(uint));
	cudaMemset(d_errNum, 0, sizeof(uint));
	
	blockSize = 512;
	numBlocks = size / (blockSize * THREAD_SIZE);
	
	sortVerifyKernel <<<numBlocks, blockSize>>>(d_DstKey, d_DstVal, d_SrcKey, d_errNum);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(&h_errNum, d_errNum, sizeof(uint), cudaMemcpyDeviceToHost);
	
	cudaFree(d_errNum);

	return h_errNum;
}
