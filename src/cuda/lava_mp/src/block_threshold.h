/*
 * block_threshold.h
 *
 *  Created on: Dec 3, 2019
 *      Author: fernando
 */

#ifndef BLOCK_THRESHOLD_H_
#define BLOCK_THRESHOLD_H_

//Assuming that max boxes will be 25
#define THRESHOLD_SIZE 1 << 14 //IT Can be up to boxes * boxes * boxes

#define THRESHOLD_PATH "/home/carol/radiation-benchmarks/data/lava/threshold.data"

__device__ uint32_t thresholds[THRESHOLD_SIZE] = { 0 };

#endif /* BLOCK_THRESHOLD_H_ */
