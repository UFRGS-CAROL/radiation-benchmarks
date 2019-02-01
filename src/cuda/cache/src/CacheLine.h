/*
 * CacheLine.h
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#ifndef CACHELINE_H_
#define CACHELINE_H_

#include "kernels.h"

//alignas(LINE_SIZE)
template<uint32 LINE_SIZE>
struct CacheLine {
	byte t[LINE_SIZE]; //byte type

	__host__ __device__ CacheLine() {
	}

	__host__ __device__ CacheLine(const CacheLine<LINE_SIZE>& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			this->t[i] = T.t[i];

		}
	}

	__host__ __device__ CacheLine(const byte& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T;
		}
	}

	inline CacheLine& operator=(const byte& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T;
		}
		return *this;
	}

	__host__ __device__ inline byte operator^(const byte& rhs) volatile {
		byte ret = rhs;
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			ret ^= t[i];
		}
		return ret;
	}

	__host__ __device__ inline bool operator !=(const byte& a) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			if (a != t[i])
				return true;
		}
		return false;
	}
};

__device__ static void sleep_cuda(std::int64_t clock_count) {

	std::int64_t start = clock64();
	std::int64_t clock_offset = 0;
	while (clock_offset < clock_count) {
		clock_offset = clock64() - start;
	}
}

#endif /* CACHELINE_H_ */
