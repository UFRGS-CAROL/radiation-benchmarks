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
#define NUMBER_THREAD_PER_BLOCK MAX_THREAD_BLOCK * 64

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

#define NUM_COMPOSE_DIVISOR 1000000
#define MUL_INPUT  1.0000001
#define FMA_INPUT 0.0005

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


#endif /* COMMON_H_ */
