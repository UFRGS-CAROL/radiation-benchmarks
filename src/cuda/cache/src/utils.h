/*
 * utils.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <cuda_runtime.h> // cudaError_t
#include <string> // error message
#include <iostream>

#include <vector>

#include "cuda_utils.h"

#define DEFAULT_INDEX -1;


//typedef unsigned char byte;
typedef unsigned int uint32;
typedef unsigned long long int uint64;

typedef int int32;
typedef long long int int64;

static void error(const std::string& err) {
	throw std::runtime_error("ERROR:" + err);
}

#endif /* UTILS_H_ */
