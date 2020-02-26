/*
 * common.h
 *
 *  Created on: 25/02/2020
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932

const int threads_per_block = 512;

typedef float float_t;

/**
 @var M value for Linear Congruential Generator (LCG); use GCC's value
 */
constexpr long M = INT_MAX;
/**
 @var A value for LCG
 */
constexpr int A = 1103515245;
/**
 @var C value for LCG
 */
constexpr int C = 12345;

#endif /* COMMON_H_ */
