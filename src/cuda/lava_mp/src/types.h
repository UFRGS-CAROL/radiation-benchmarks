/*
 * three_vector.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef THREE_VECTOR_H_
#define THREE_VECTOR_H_

#include <vector>
#include "include/device_vector.h"
#include "include/cuda_utils.h"

/**
 * ABS
 */
__DEVICE_HOST_INLINE__
double abs__(double a) {
	return fabs(a);
}

__DEVICE_HOST_INLINE__
float abs__(float a) {
	return fabsf(a);
}

__DEVICE_HOST_INLINE__
half abs__(half a) {
	return fabsf(a);
}

template<typename real_t>
real_t abs__(real_t& lhs);

template<typename tested_type>
struct THREE_VECTOR {
	tested_type x, y, z;
};

template<typename tested_type>
struct FOUR_VECTOR {
	tested_type v, x, y, z;

	__DEVICE_HOST_INLINE__
	bool operator==(const FOUR_VECTOR& rhs) const {
		return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z)
				&& (this->v == rhs.v);
	}

	__DEVICE_HOST_INLINE__
	bool operator!=(const FOUR_VECTOR& rhs) const {
		if ((abs__(this->v - tested_type(rhs.v)) > ZERO_DOUBLE) ||	//V
			(abs__(this->x - tested_type(rhs.x)) > ZERO_DOUBLE) ||	//X
			(abs__(this->y - tested_type(rhs.y)) > ZERO_DOUBLE) ||	//Y
			(abs__(this->z - tested_type(rhs.z)) > ZERO_DOUBLE)) {	//Z
			return true;
		}
		return false;
	}

	friend std::ostream& operator<<(std::ostream& os, const FOUR_VECTOR& lhs) {
		os << lhs.v << " " << lhs.x << " " << lhs.y << " " << lhs.z;
		return os;
	}

	__DEVICE_HOST_INLINE__
	FOUR_VECTOR<tested_type>& operator=(const FOUR_VECTOR<double>& rhs) {
		this->v = double(rhs.v);
		this->x = double(rhs.x);
		this->y = double(rhs.y);
		this->z = double(rhs.z);
		return *this;
	}

	__DEVICE_HOST_INLINE__
	FOUR_VECTOR<tested_type>& operator=(const FOUR_VECTOR<float>& rhs) {
		this->v = float(rhs.v);
		this->x = float(rhs.x);
		this->y = float(rhs.y);
		this->z = float(rhs.z);
		return *this;
	}
};

template<typename tested_type>
struct par_str {
	tested_type alpha;
};

struct dim_str {
	// input arguments
	int cur_arg;
	int arch_arg;
	int cores_arg;
	int boxes1d_arg;
	// system memory
	long number_boxes;
	long box_mem;
	long space_elem;
	long space_mem;
	long space_mem2;
};

struct nei_str {
	// neighbor box
	int x, y, z;
	int number;
	long offset;
};

struct box_str {
	// home box
	int x, y, z;
	int number;
	long offset;
	// neighbor boxes
	int nn;
	nei_str nei[26];

	__DEVICE_HOST_INLINE__
	bool operator==(const box_str& rhs) {
		return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z)
				&& (this->number == rhs.number) && (this->offset == rhs.offset);
	}
	__DEVICE_HOST_INLINE__
	bool operator!=(const box_str& rhs) {
		return !(*this == rhs);
	}
};

template<typename T> using VectorOfDeviceVector = std::vector<rad::DeviceVector<T>>;

struct CudaStream {
	cudaStream_t stream;

	CudaStream() {
		rad::checkFrameworkErrors(
				cudaStreamCreateWithFlags(&this->stream,
						cudaStreamNonBlocking));
	}

	~CudaStream() {
		rad::checkFrameworkErrors(cudaStreamDestroy(this->stream));
	}

	void sync() {
		rad::checkFrameworkErrors(cudaStreamSynchronize(this->stream));
	}
};

#endif /* THREE_VECTOR_H_ */
