/*
 * types.h
 *
 *  Created on: 22/05/2019
 *      Author: fernando
 */

#ifndef TYPES_H_
#define TYPES_H_

//=============================================================================
//	STRUCTURES
//=============================================================================

template<typename T>
struct THREE_VECTOR {
	T x, y, z;
};

template<typename T>
struct FOUR_VECTOR {
	T v, x, y, z;

	__host__ __device__ inline  bool operator==(const FOUR_VECTOR& rhs) {
		return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z)
				&& (this->v == rhs.v);
	}

	__host__ __device__ inline  bool operator!=(const FOUR_VECTOR& rhs) {
		return !(*this == rhs);
	}

};

//template<typename T>
//struct THREE_VECTOR_HOST {
//	T x, y, z;
//};

//template<typename T>
//struct FOUR_VECTOR_HOST {
//	T v, x, y, z;
//
//
//
//};


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

	__host__ __device__ inline bool operator==(const box_str& rhs) {
		return (this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z)
				&& (this->number == rhs.number) && (this->offset == rhs.offset);
	}
	__host__ __device__ inline bool operator!=(const box_str& rhs) {
		return !(*this == rhs);
	}

} ;

template<typename T>
struct par_str {
	T alpha;
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

#endif /* TYPES_H_ */
