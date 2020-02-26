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

#define BLOCK_SIZE 1024 //256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration
#define BENCH_PRINT
#define M_SEED 9
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))


#include "device_vector.h"
#include "cuda_utils.h"

struct cuda_stream {
	cudaStream_t stream;
	cuda_stream() {
		rad::checkFrameworkErrors(cudaStreamCreate(&(this->stream)));
	}
	~cuda_stream() {
		rad::checkFrameworkErrors(cudaStreamDestroy(this->stream));
	}
	cudaStream_t operator*(){
		return this->stream;
	}
};


template<typename T>
using vector = std::vector<T>;

template<typename T>
using matrix_hst = vector<vector<T>>;

template<typename T>
using matrix_dev = vector<rad::DeviceVector<T>>;

int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols,
		int pyramid_height, int blockCols, int borderCols, cuda_stream& stream);

#endif /* COMMON_H_ */
