/*
 * common.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <unordered_map>

#define __DEVICE_INLINE__ __device__ __forceinline__
#define __DEVICE_HOST_INLINE__ __device__ __host__ __forceinline__

// MIN percent
#define MIN_PERCENTAGE 0.99f

// MAX percent
#define MAX_PERCENTAGE 1.001f

/**
 =============================================================================
 DEFINE / INCLUDE
 =============================================================================
 keep this low to allow more blocks that share shared memory to run concurrently,
 code does not work for larger than 110, more speedup can be achieved with
 larger number and no shared memory used
 **/
#define NUMBER_PAR_PER_BOX 192

// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance
#define NUMBER_THREADS 192

// STABLE
#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))
#define MAX_LOGGED_ERRORS_PER_STREAM 100

//Subtraction considering the signal
#define SUB_ABS(lhs, rhs) ((lhs > rhs) ? (lhs - rhs) : (rhs - lhs))
#define ABSF(lhs, rhs) (fabsf(lhs - rhs))

#ifndef ZERO_FLOAT
#define ZERO_FLOAT 2.2e-20
#endif

#ifndef ZERO_DOUBLE
#define ZERO_DOUBLE 1.4e-60
#endif

#ifndef ZERO_HALF
#define ZERO_HALF 9E-04
#endif

static void __error(std::string err, const char* file, const int line) {
	throw std::runtime_error(
			"ERROR:" + err + " at file:" + std::string(file) + " at line:"
					+ std::to_string(line));
}

#define error(str) __error(str, __FILE__, __LINE__)

typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

typedef enum {
	NONE, DMR, TMR, DMRMIXED
} REDUNDANCY;

static std::unordered_map<std::string, REDUNDANCY> RED = {
//NONE
		{ "none", NONE },
		//DMR
		{ "dmr", DMR },
		// DMRMIXED
		{ "dmrmixed", DMRMIXED }, };

static std::unordered_map<std::string, PRECISION> PRE = {
//HALF
		{ "half", HALF },
		//SINGLE
		{ "single", SINGLE },
		// DOUBLE
		{ "double", DOUBLE }, };

#endif /* COMMON_H_ */
