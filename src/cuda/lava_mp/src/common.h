/*
 * common.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

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

#endif /* COMMON_H_ */
