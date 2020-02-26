
#include "common.h"

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc,
		int *gpuResults, int cols, int rows, int startStep, int border) {

	__shared__ int prev[BLOCK_SIZE];
	__shared__ int result[BLOCK_SIZE];

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	// each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data

	// calculate the small block size
	int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkX = small_block_cols * bx - border;
	int blkXmax = blkX + BLOCK_SIZE - 1;

	// calculate the global thread coordination
	int xidx = blkX + tx;

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > cols - 1) ?
	BLOCK_SIZE - 1 - (blkXmax - cols + 1) :
											BLOCK_SIZE - 1;

	int W = tx - 1;
	int E = tx + 1;

	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if (IN_RANGE(xidx, 0, cols - 1)) {
		prev[tx] = gpuSrc[xidx];
	}
	__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
	bool computed;
	for (int i = 0; i < iteration; i++) {
		computed = false;
		if ( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid) {
			computed = true;
			int left = prev[W];
			int up = prev[tx];
			int right = prev[E];
			int shortest = MIN(left, up);
			shortest = MIN(shortest, right);
			int index = cols * (startStep + i) + xidx;
			result[tx] = shortest + gpuWall[index];

		}
		__syncthreads();
		if (i == iteration - 1)
			break;
		if (computed)	 //Assign the computation range
			prev[tx] = result[tx];
		__syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on ``computed''
	if (computed) {
		gpuResults[xidx] = result[tx];
	}
}

/*
 compute N time steps
 */
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
		int pyramid_height, int blockCols, int borderCols) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(blockCols);

	int src = 1, dst = 0;
	for (int t = 0; t < rows - 1; t += pyramid_height) {
		int temp = src;
		src = dst;
		dst = temp;
		dynproc_kernel<<<dimGrid, dimBlock>>>(MIN(pyramid_height, rows - t - 1),
				gpuWall, gpuResult[src], gpuResult[dst], cols, rows, t,
				borderCols);
	}
	return dst;
}
