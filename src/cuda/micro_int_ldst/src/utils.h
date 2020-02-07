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

//Max number of load/stores performed
//each time
#define MEM_OPERATION_NUM 64u

//the size of the random input array
//for load/store benchmark, and for arithmetic int
#define DEFAULT_INPUT_ARRAY 1024u

//for load and store the OPS value is not used, then we use
//max thread ld/st operation
#define MAX_THREAD_LD_ST_OPERATIONS (DEFAULT_INPUT_ARRAY / MEM_OPERATION_NUM)

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
