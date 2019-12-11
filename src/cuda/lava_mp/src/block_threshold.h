/*
 * block_threshold.h
 *
 *  Created on: Dec 3, 2019
 *      Author: fernando
 */

#ifndef BLOCK_THRESHOLD_H_
#define BLOCK_THRESHOLD_H_

//Assuming that max boxes will be 25

#define THRESHOLD_SIZE_THREAD (23 * 23 * 23 * NUMBER_THREADS) //works only for specific size

#define THRESHOLD_SIZE THRESHOLD_SIZE_THREAD
//int(1 << 14) //IT Can be up to boxes * boxes * boxes


// one day I will fix this, but not today
#define THRESHOLD_PATH "/home/carol/radiation-benchmarks/data/lava/threshold.data"

__device__ uint32_t thresholds[THRESHOLD_SIZE] = { 0 };

__device__ float lower_relative_limit[THRESHOLD_SIZE_THREAD];
__device__ float upper_relative_limit[THRESHOLD_SIZE_THREAD];


#endif /* BLOCK_THRESHOLD_H_ */
