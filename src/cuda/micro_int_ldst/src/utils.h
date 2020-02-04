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
