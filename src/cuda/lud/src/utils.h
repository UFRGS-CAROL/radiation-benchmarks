/*
 * utils.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <string> // error message
#include <stdexcept>


#define RANGE_INT_VAL 100

static inline void __throw_line(std::string err, std::string line, std::string file) {
	throw std::runtime_error(err + " at " + file + ":" + line);

}

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));

#define BLOCK_SIZE 32


#endif /* UTILS_H_ */
