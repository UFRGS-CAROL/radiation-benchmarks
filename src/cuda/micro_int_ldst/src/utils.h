/*
 * utils.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string> // error message
#include <unordered_map> // micro selection

#define DEFAULT_INDEX -1;

#define WARP_SIZE 32ull
#define WARP_PER_SM 4ull

#define MAX_THREAD_BLOCK WARP_SIZE * WARP_SIZE

#define RANGE_INT_VAL 100


//The amount of memory that will be used
//in the LDST test
//Default is 1GB
#define GPU_DDR_TEST_SIZE (1024ull * 1024ull * 1024ull)

//Operations performed by each thread
//Max tested that worked is 64
#define MEM_OPERATION_NUM 64

//for load and store the OPS value is not used, then we use
//max thread ld/st operation
#define MAX_THREAD_LD_ST_OPERATIONS 8

//availiable cores * THREAD_MULTIPLIER
#define THREAD_MULTIPLIER 8

#ifndef OPS
#define OPS 10000000
#endif

typedef enum {
	ADD_INT, MUL_INT, MAD_INT, LDST
} MICROINSTRUCTION;


 static std::unordered_map<std::string, MICROINSTRUCTION> mic = {
//ADD
		{ "add", ADD_INT },
		//MUL
		{ "mul", MUL_INT },
		//FMA
		{ "mad", MAD_INT },

		{ "ldst", LDST},
		};


void __throw_line(std::string err, std::string line, std::string file);

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

#endif /* UTILS_H_ */
