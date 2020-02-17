/*
 * utils.h
 *
 *  Created on: 15/02/2020
 *      Author: fernando
 */

#ifndef UTILS_H_
#define UTILS_H_

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else

#define MAXBLOCKSIZE 1024 //512
#endif

//2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

//it is the decimal places for
//logging errors, 20 is from old benchmarks
#define ERROR_LOG_PRECISION 20

#include "device_vector.h"

static inline void __throw_line(std::string err, std::string line, std::string file) {
	throw std::runtime_error(err + " at " + file + ":" + line);
}

#define throw_line(err) __throw_line(std::string(err), std::to_string(__LINE__), std::string(__FILE__));


void ForwardSub(rad::DeviceVector<float>& m_cuda,
		rad::DeviceVector<float>& a_cuda, rad::DeviceVector<float>& b_cuda,
		size_t size);

void BackSub(std::vector<float>& finalVec, std::vector<float>& a,
		std::vector<float>& b, size_t size);
#endif /* UTILS_H_ */
