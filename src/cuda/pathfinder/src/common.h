/*
 * common.h
 *
 *  Created on: 24/02/2020
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cstdint>

#define MAX_THREADS_PER_BLOCK 1024 //512

#define BLOCK_SIZE 1024 //256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration
#define BENCH_PRINT
#define M_SEED 9
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


void run(int argc, char** argv);


#endif /* COMMON_H_ */
