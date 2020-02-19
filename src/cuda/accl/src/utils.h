/*
 * utils.h
 *
 *  Created on: 15/02/2020
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_



//it is the decimal places for
//logging errors, 20 is from old benchmarks
#define ERROR_LOG_PRECISION 20

#include "device_vector.h"

static inline void __throw_line(std::string err, std::string line, std::string file) {
	throw std::runtime_error(err + " at " + file + ":" + line);
}

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

#endif /* UTILS_H_ */
