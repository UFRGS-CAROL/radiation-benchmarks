/*
 * CacheLine.h
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#ifndef CACHELINE_H_
#define CACHELINE_H_

#include <ostream>
#include "utils.h"

template<uint32 LINE_SIZE>
struct
//alignas(LINE_SIZE)
CacheLine {
	volatile byte t[LINE_SIZE]; //byte type

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

	__host__ __device__ inline CacheLine& operator=(const byte& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T;
		}
		return *this;
	}

	__host__ __device__ inline volatile CacheLine& operator=(
			CacheLine<LINE_SIZE>& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T.t[i];
		}
		return *this;
	}

	__host__ __device__ inline CacheLine& operator=(
			const CacheLine<LINE_SIZE>& T) {
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T.t[i];
		}
		return *this;
	}

//	__host__ __device__ inline CacheLine& operator=(register volatile CacheLine<LINE_SIZE>& T) {
//#pragma unroll
//		for (int i = 0; i < LINE_SIZE; i++) {
//			t[i] = T.t[i];
//		}
//		return *this;
//	}

	__host__ __device__ inline byte operator^(const byte& rhs) {
		byte ret = rhs;
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			ret ^= t[i];
		}
		return ret;
	}

	__host__ __device__ inline bool operator !=(const byte& a) volatile{
#pragma unroll
		for (int i = 0; i < LINE_SIZE; i++) {
			if (a != t[i])
				return true;
		}
		return false;
	}

	__host__ friend std::ostream& operator<<(std::ostream& stream,
			const CacheLine<LINE_SIZE>& t) {
		for (auto s : t.t) {
			stream << " " << s;
		}
		return stream;
	}

	__host__ __device__ byte operator [](int idx) const {
		return t[idx];
	}
};


#endif /* CACHELINE_H_ */
