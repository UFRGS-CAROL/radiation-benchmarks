/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */

#include <vector>
#include <cuda.h>

#include "device_vector.h"
#include "cuda_utils.h"
#include "utils.h"

/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties() {
	cudaDeviceProp deviceProp;
	int nDevCount = 0;

	cudaGetDeviceCount(&nDevCount);
	printf("Total Device found: %d", nDevCount);
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx) {
		memset(&deviceProp, 0, sizeof(deviceProp));
		if (cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx)) {
			printf("\nDevice Name \t\t - %s ", deviceProp.name);
			printf("\n**************************************");
			printf("\nTotal Global Memory\t\t\t - %lu KB",
					deviceProp.totalGlobalMem / 1024);
			printf("\nShared memory available per block \t - %lu KB",
					deviceProp.sharedMemPerBlock / 1024);
			printf("\nNumber of registers per thread block \t - %d",
					deviceProp.regsPerBlock);
			printf("\nWarp size in threads \t\t\t - %d", deviceProp.warpSize);
			printf("\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch);
			printf("\nMaximum threads per block \t\t - %d",
					deviceProp.maxThreadsPerBlock);
			printf("\nMaximum Thread Dimension (block) \t - %d %d %d",
					deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
					deviceProp.maxThreadsDim[2]);
			printf("\nMaximum Thread Dimension (grid) \t - %d %d %d",
					deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
					deviceProp.maxGridSize[2]);
			printf("\nTotal constant memory \t\t\t - %zu bytes",
					deviceProp.totalConstMem);
			printf("\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major,
					deviceProp.minor);
			printf("\nClock rate \t\t\t\t - %d KHz", deviceProp.clockRate);
			printf("\nTexture Alignment \t\t\t - %zu bytes",
					deviceProp.textureAlignment);
			printf("\nDevice Overlap \t\t\t\t - %s",
					deviceProp.deviceOverlap ? "Allowed" : "Not Allowed");
			printf("\nNumber of Multi processors \t\t - %d\n\n",
					deviceProp.multiProcessorCount);
		} else
			printf("\n%s", cudaGetErrorString(cudaGetLastError()));
	}
}


/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun(std::vector<float>& m) {
	for (auto& mi : m)
		mi = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t) {
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
		return;
	*(m_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) =
			*(a_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1)
					+ t) / *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size,
		int j1, int t) {
	if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
		return;
	if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t)
		return;

	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

	a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -= m_cuda[Size * (xidx + 1 + t)
			+ t] * a_cuda[Size * t + (yidx + t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if (yidx == 0) {
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx + 1 + t] -= m_cuda[Size * (xidx + 1 + t) + (yidx + t)]
				* b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
template<typename real_t>
void ForwardSub(std::vector<real_t>& m, std::vector<real_t>& a,
		std::vector<real_t>& b, size_t size, float& totalKernelTime) {
	int t;
	size_t matrix_size = size * size;

	// allocate memory on GPU
	// copy memory to GPU
	rad::DeviceVector<real_t> m_cuda(m);
	rad::DeviceVector<real_t> a_cuda(a);
	rad::DeviceVector<real_t> b_cuda(b);

//	int block_size, grid_size;

	size_t block_size = MAXBLOCKSIZE;
	size_t grid_size = (size / block_size) + (!(size % block_size) ? 0 : 1);
	//printf("1d grid size: %d\n",grid_size);

	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );

	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (size / blockSize2d) + (!(size % blockSize2d ? 0 : 1));

	dim3 dimBlockXY(blockSize2d, blockSize2d);
	dim3 dimGridXY(gridSize2d, gridSize2d);

	// begin timing kernels
	auto time_start = rad::mysecond();

	for (t = 0; t < (size - 1); t++) {
		Fan1<<<dimGrid, dimBlock>>>(m_cuda.data(), a_cuda.data(), size, t);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		Fan2<<<dimGridXY, dimBlockXY>>>(m_cuda.data(), a_cuda.data(), b_cuda.data(),
				size, size - t, t);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
		;
	}

	auto time_end = rad::mysecond();
	totalKernelTime = time_end - time_start;
	// copy memory back to CPU
	m_cuda.to_vector(m);
	a_cuda.to_vector(a);
	b_cuda.to_vector(b);
}

void ForwardSub(std::vector<float>& m, std::vector<float>& a,
		std::vector<float>& b, size_t size, float& totalKernelTime) {
	ForwardSub(m, a, b, size, totalKernelTime);
}


/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub(std::vector<float>& finalVec, std::vector<float>& a,
		std::vector<float>& b, unsigned Size) {
	// solve "bottom up"
	int i, j;
	for (i = 0; i < Size; i++) {
		finalVec[Size - i - 1] = b[Size - i - 1];
		for (j = 0; j < i; j++) {
			finalVec[Size - i - 1] -= *(a.data() + Size * (Size - i - 1)
					+ (Size - j - 1)) * finalVec[Size - j - 1];
		}
		finalVec[Size - i - 1] = finalVec[Size - i - 1]
				/ *(a.data() + Size * (Size - i - 1) + (Size - i - 1));
	}
}

