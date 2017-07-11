/*
 * cudaUtil.cpp
 *
 *  Created on: Jun 6, 2017
 *      Author: carol
 */

#include "cudaUtil.h"

//1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
void cuda_gridsize(dim3 *threads, dim3 *blocks, size_t x, size_t y,
		size_t z) {
	int true_block_size = BLOCK_SIZE;
	if(y == 1 && z == 1)
		true_block_size = BLOCK_SIZE * BLOCK_SIZE;


	long blocks_x = ceil(float(x) / float(true_block_size));
	long threads_x = ceil(float(x) / float(blocks_x));
	long blocks_y = ceil(float(y) / float(true_block_size));
	long threads_y = ceil(float(y) / float(blocks_y));
	long blocks_z = ceil(float(z) / float(true_block_size));
	long threads_z = ceil(float(z) / float(blocks_z));

	*blocks = dim3(blocks_x, blocks_y, blocks_z);
	*threads = dim3(threads_x, threads_y, threads_z);

//	printf("b_x %d b_y %d b_z %d\nt_x %d t_y %d t_z %d\n", blocks->x, blocks->y,
//			blocks->z, threads->x, threads->y, threads->z);

}
