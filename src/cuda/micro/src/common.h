/*
 * common.h
 *
 *  Created on: Mar 15, 2020
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <string> // error message
#include <unordered_map> // micro selection

#define DEFAULT_INDEX -1;

#define WARP_SIZE 32ull
#define WARP_PER_SM 4ull

#define MAX_THREAD_BLOCK WARP_SIZE * WARP_SIZE

#define RANGE_INT_MAX 1024
#define RANGE_INT_MIN 3

#define LOOPING_UNROLL 128

typedef enum {
	ADD, MUL, FMA, MAD, DIV, PYTHAGOREAN, EULER, LDST, BRANCH
} MICROINSTRUCTION;

typedef enum {
	HALF, SINGLE, DOUBLE, INT32, INT64
} PRECISION;

#endif /* COMMON_H_ */
