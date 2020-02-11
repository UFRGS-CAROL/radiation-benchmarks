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

#define OPS 10000
#define LOOPING_UNROLL 512

typedef enum {
	ADD, MUL, FMA, PYTHAGOREAN, EULER, LOG
} MICROINSTRUCTION;


typedef enum {
	HALF, SINGLE, DOUBLE
} PRECISION;

static std::unordered_map<std::string, MICROINSTRUCTION> mic = {
//ADD
		{ "add", ADD },
		//MUL
		{ "mul", MUL },
		//FMA
		{ "fma", FMA },

		//Pythagorean
		{ "pythagorean", PYTHAGOREAN },

		//EULER
		{ "euler", EULER },

		//Log
		{ "log", LOG }, };

static std::unordered_map<std::string, PRECISION> pre = {
//half
		{ "half", HALF },
		//float
		{ "single", SINGLE },
		{ "float", SINGLE },
		//double
		{ "double", DOUBLE },
};


static void __throw_line(std::string err, std::string line, std::string file){
    throw std::runtime_error("ERROR at " + file + ":" + line);
}


#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

#endif /* UTILS_H_ */
