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
#include <vector>

#define __CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__

#define INT_TYPE uint32
#define CHUNK_SIZE(line_n) (line_n / sizeof(INT_TYPE))

template<uint32 LINE_SIZE>
struct CacheLine {
	//volatile
	uint32 t[LINE_SIZE]; //byte type

	__CUDA_HOST_DEVICE__ CacheLine() {
	}

	__CUDA_HOST_DEVICE__ CacheLine(const CacheLine& T) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
			this->t[i] = T.t[i];
		}
	}

	__CUDA_HOST_DEVICE__ CacheLine(const byte& T) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
//			t[i] = T;
			this->set_byte(T, this->t + i);
		}
	}


	__CUDA_HOST_DEVICE__ void set_byte(const byte& b, uint32* it){
		((byte*)it)[0] = b;
		((byte*)it)[1] = b;
		((byte*)it)[2] = b;
		((byte*)it)[3] = b;
	}

	__CUDA_HOST_DEVICE__ CacheLine& operator=(const byte& T) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
//			t[i] = T;
			this->set_byte(T, this->t + i);
		}
		return *this;
	}

	__CUDA_HOST_DEVICE__ CacheLine& operator=(
			CacheLine& T) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
			t[i] = T.t[i];
		}
		return *this;
	}

	__CUDA_HOST_DEVICE__ CacheLine& operator=(
			const CacheLine& T) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
			t[i] = T.t[i];
		}
		return *this;
	}

//	__CUDA_HOST_DEVICE__ byte operator^(const byte& rhs) {
//		byte ret = rhs;
//#pragma unroll
//		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
//			ret ^= t[i];
//		}
//		return ret;
//	}

	__CUDA_HOST_DEVICE__ CacheLine& operator&=(const CacheLine& rhs) {
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
			this->t[i] &= rhs.t[i];
		}
		return *this;
	}

	__CUDA_HOST_DEVICE__ bool operator !=(const byte& a)  {
		uint32 a_cpy;

		this->set_byte(a, &a_cpy);
#pragma unroll
		for (int i = 0; i < CHUNK_SIZE(LINE_SIZE); i++) {
			if (a_cpy != t[i])
				return true;
		}
		return false;
	}

	__host__ friend std::ostream& operator<<(std::ostream& stream,
			const CacheLine& t) {
		for (auto s : t.t) {
			stream << " " << s;
		}
		return stream;
	}

	__CUDA_HOST_DEVICE__ byte operator [](int idx) const {
		return t[idx];
	}
};

#endif /* CACHELINE_H_ */
