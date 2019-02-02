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

void check_cuda_error_(const char *file, unsigned line, const char *statement,
		cudaError_t err);

#define cuda_check(value) check_cuda_error_(__FILE__,__LINE__, #value, value)

void error(std::string err);
void sleep(int seconds);

size_t get_time_since_epoch();



#endif /* UTILS_H_ */
