/*
 * common.h
 *
 *  Created on: 24/02/2020
 *      Author: fernando
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <cstdint>

#define MAX_THREADS_PER_BLOCK 1024 //512

typedef uint8_t bool_t;

enum {
	FALSE = 0, TRUE = 1
};

//Structure to hold a node information
struct Node {
	int starting;
	int no_of_edges;
};

#include "device_vector.h"
#include <vector>


template<typename T>
using DevMat = std::vector<rad::DeviceVector<T>>;

#endif /* COMMON_H_ */
