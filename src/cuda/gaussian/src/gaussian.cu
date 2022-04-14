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

#include "cuda_utils.h"
#include "utils.h"
#include "multi_compiler_analysis.h"

std::string get_multi_compiler_header() {
	return rad::get_multi_compiler_header();
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
	*(m_cuda + Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t) = *(a_cuda
			+ Size * (blockDim.x * blockIdx.x + threadIdx.x + t + 1) + t)
			/ *(a_cuda + Size * t + t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t) {
	if (threadIdx.x + blockIdx.x * blockDim.x >= Size - 1 - t)
		return;
	if (threadIdx.y + blockIdx.y * blockDim.y >= Size - t)
		return;

	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

	a_cuda[Size * (xidx + 1 + t) + (yidx + t)] -= m_cuda[Size * (xidx + 1 + t) + t]
			* a_cuda[Size * t + (yidx + t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if (yidx == 0) {
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx + 1 + t] -= m_cuda[Size * (xidx + 1 + t) + (yidx + t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
template<typename real_t>
void ForwardSubTemplate(rad::DeviceVector<real_t>& m_cuda, rad::DeviceVector<real_t>& a_cuda,
		rad::DeviceVector<real_t>& b_cuda, size_t size) {
	// allocate memory on GPU
	// copy memory to GPU
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

	for (size_t t = 0; t < (size - 1); t++) {
		Fan1<<<dimGrid, dimBlock>>>(m_cuda.data(), a_cuda.data(), size, t);
		rad::checkFrameworkErrors (cudaDeviceSynchronize());;
		Fan2<<<dimGridXY, dimBlockXY>>>(m_cuda.data(), a_cuda.data(), b_cuda.data(), size, size - t,
				t);
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors (cudaGetLastError());;
	}
}

void ForwardSub(rad::DeviceVector<float>& m_cuda, rad::DeviceVector<float>& a_cuda,
		rad::DeviceVector<float>& b_cuda, size_t size) {
	ForwardSubTemplate(m_cuda, a_cuda, b_cuda, size);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub(std::vector<float>& finalVec, std::vector<float>& a, std::vector<float>& b,
		size_t size) {
	// solve "bottom up"
	for (size_t i = 0; i < size; i++) {
		finalVec[size - i - 1] = b[size - i - 1];
		for (size_t j = 0; j < i; j++) {
			finalVec[size - i - 1] -= a[size * (size - i - 1) + (size - j - 1)]
					* finalVec[size - j - 1];
		}
		finalVec[size - i - 1] = finalVec[size - i - 1] / a[size * (size - i - 1) + (size - i - 1)];
	}
}

