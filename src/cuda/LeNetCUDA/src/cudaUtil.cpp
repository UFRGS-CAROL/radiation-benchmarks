/*
 * cudaUtil.cpp
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */




//1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
void cuda_gridsize(size_t x, size_t y = 1, size_t z = 1, dim3 *threads, dim3 *blocks){

	long blocks_x = ceil(float(x) / float(BLOCK_SIZE));
	long threads_x = ceil(float(x) / float(blocks_x));
	long blocks_y = ceil(float(y) / float(BLOCK_SIZE));
	long threads_y = ceil(float(y) / float(blocks_y));
	long blocks_z = ceil(float(z) / float(BLOCK_SIZE));
	long threads_z = ceil(float(z) / float(blocks_z));

	blocks = dim3(blocks_x, blocks_y, blocks_z);
	threads = dim3(threads_x, threads_y, threads_z);

}
