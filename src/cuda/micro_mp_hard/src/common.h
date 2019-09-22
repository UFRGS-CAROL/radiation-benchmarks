/*
 * common.h
 *
 *  Created on: 16/09/2019
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#define BLOCK_SIZE 32
#define MAX_THREAD_BLOCK BLOCK_SIZE * BLOCK_SIZE
#define NUMBER_OF_THREAD_BLOCK MAX_THREAD_BLOCK * 128

#define PRECISION_PLACES 20

typedef enum {
	ADD, MUL, FMA, ADDNOTBIASED, MULNOTBIASED, FMANOTBIASED
} MICROINSTRUCTION;

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED, TMRMIXED
} REDUNDANCY;

typedef uint64_t uint64;
typedef uint32_t uint32;
typedef unsigned char byte;

#ifndef ZERO_FLOAT
#define ZERO_FLOAT 2.2e-07
#endif

#ifndef ZERO_HALF
#define ZERO_HALF 4.166E-05
#endif

#ifndef ZERO_FULL
#define ZERO_FULL 0.0L
#endif

#define __DEVICE_HOST__ __device__ __host__ __forceinline__
#define __HOST__ __host__ __forceinline__
#define __DEVICE__ __device__ __forceinline__

#ifndef MAXSHAREDMEMORY
#define MAXSHAREDMEMORY 48 * 1024
#endif

#define VOLTA_BLOCK_MULTIPLIER 40.9f

std::unordered_map<std::string, REDUNDANCY> red = {
//NONE
		{ "none", NONE },
		//DMR
		{ "dmr", DMR },
		// DMRMIXED
		{ "dmrmixed", DMRMIXED },
//TMRMIXED
//         {"TMRMIXED",  XAVIER}
		};

std::unordered_map<std::string, PRECISION> pre = {
//HALF
		{ "half", HALF },
		//SINGLE
		{ "single", SINGLE },
		// DOUBLE
		{ "double", DOUBLE }, };

std::unordered_map<std::string, MICROINSTRUCTION> mic = {
//ADD
		{ "add", ADD },
		//MUL
		{ "mul", MUL },
		//FMA
		{ "fma", FMA }, };

/**
 * Define the threshold to use on
 * the comparison method
 */
//For 1 iteration
#define ADD_UINT32_THRESHOLD_1 5
#define MUL_UINT32_THRESHOLD_1 11
#define FMA_UINT32_THRESHOLD_1 10

//For 10 iterations
#define ADD_UINT32_THRESHOLD_10 5
#define MUL_UINT32_THRESHOLD_10 22
#define FMA_UINT32_THRESHOLD_10 10

//For 100 iterations
#define ADD_UINT32_THRESHOLD_100 50
#define MUL_UINT32_THRESHOLD_100 190
#define FMA_UINT32_THRESHOLD_100 49

//For 1K iterations
#define ADD_UINT32_THRESHOLD_1000 246
#define MUL_UINT32_THRESHOLD_1000 965
#define FMA_UINT32_THRESHOLD_1000 247


#endif /* COMMON_H_ */
