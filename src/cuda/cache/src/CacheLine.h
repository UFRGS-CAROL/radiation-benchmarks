/*
 * CacheLine.h
 *
 *  Created on: Jan 31, 2019
 *      Author: carol
 */

#ifndef CACHELINE_H_
#define CACHELINE_H_

typedef std::uint8_t byte;

//alignas(LINE_SIZE)
template<std::uint32_t LINE_SIZE>
struct CacheLine {
	byte t[LINE_SIZE]; //byte type

//	CacheLine() {
//	}
//
//	CacheLine(const CacheLine& a) {
//		for (int i = 0; i < LINE_SIZE; i++)
//			t[i] = a.t[i];
//	}
//
//	inline CacheLine& operator=(const CacheLine& a) {
//		t = a.t;
//		return *this;
//	}
	inline CacheLine& operator=(const byte& T) {
		for (int i = 0; i < LINE_SIZE; i++) {
			t[i] = T;
		}
		return *this;
	}
//
//	inline bool operator==(const CacheLine& a) {
//		for (int i = 0; i < LINE_SIZE; i++) {
//			if (a.t[i] != t[i])
//				return false;
//		}
//		return true;
//	}
//
//	inline bool operator!=(const CacheLine& a) {
//		for (int i = 0; i < LINE_SIZE; i++) {
//			if (a.t[i] != t[i])
//				return true;
//		}
//		return false;
//	}
//
//	inline bool operator!=(const T a) {
//		for (int i = 0; i < LINE_SIZE; i++) {
//			if (a != t[i])
//				return true;
//		}
//		return false;
//	}
//
//	inline CacheLine operator^(const CacheLine& rhs) {
//		CacheLine ret;
//		for (int i = 0; i < LINE_SIZE; i++) {
//			ret.t[i] = t[i] ^ rhs.t[i];
//		}
	__host__ __device__ inline byte operator^(const byte& rhs) {
		byte ret = rhs;
		for (int i = 0; i < LINE_SIZE; i++) {
			ret ^= t[i];
		}
		return ret;
	}
};



__device__ void sleep_cuda(std::int64_t clock_count) {

	std::int64_t start = clock64();
	std::int64_t clock_offset = 0;
	while (clock_offset < clock_count) {
		clock_offset = clock64() - start;
	}
}


#endif /* CACHELINE_H_ */
