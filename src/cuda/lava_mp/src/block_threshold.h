/*
 * block_threshold.h
 *
 *  Created on: Dec 3, 2019
 *      Author: fernando
 */

#ifndef BLOCK_THRESHOLD_H_
#define BLOCK_THRESHOLD_H_

#define THRESHOLD_SIZE 12167
#define THRESHOLD_PATH "/home/carol/radiation-benchmarks/data/threshold.data"

__device__ uint32_t thresholds[THRESHOLD_SIZE] = { 0 };

#endif /* BLOCK_THRESHOLD_H_ */
