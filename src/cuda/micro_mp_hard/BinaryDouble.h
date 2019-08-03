/*
 * BinaryDouble.h
 *
 *  Created on: Jul 30, 2019
 *      Author: fernando
 */

#ifndef BINARYDOUBLE_H_
#define BINARYDOUBLE_H_

#include <assert.h>
#include "Parameters.h"

struct BinaryDouble {
	uint64 bin;

	__DEVICE_HOST__ uint64 double_to_uint64(const double d){
		const double* ptr = &d;
		const uint64* ptr_i = (const uint64*) ptr;
		return *ptr_i;
	}

	__DEVICE_HOST__ double uint64_to_double(const uint64 u){
		const uint64* ptr = &u;
		const double* ptr_d = (const double*) ptr;
		return *ptr_d;
	}

	__DEVICE_HOST__ operator double() {
		return this->uint64_to_double(this->bin);
	}

	__DEVICE_HOST__ BinaryDouble operator&(const BinaryDouble& r) {
		return BinaryDouble(r.bin & this->bin);
	}

	__DEVICE_HOST__ BinaryDouble operator&(const uint64& r) {
		return BinaryDouble(r & this->bin);
	}

	__DEVICE_HOST__ bool operator!=(const uint64& r) {
		return r != this->bin;
	}

	__DEVICE_HOST__ bool operator==(const uint64& r) {
		return r == this->bin;
	}


	__DEVICE_HOST__ BinaryDouble operator^(const BinaryDouble& r) {
		return BinaryDouble(r.bin ^ this->bin);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryDouble& r) {
		assert(sizeof(double) == sizeof(uint64));

		for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
			if (r.bin & i) {
				os << 1;
			} else {
				os << 0;
			}
		}

		return os;
	}

	__DEVICE_HOST__ BinaryDouble& operator=(const double& val) {
		assert(sizeof(double) == sizeof(uint64));
		this->bin = this->double_to_uint64(val);
		return *this;
	}

	__DEVICE_HOST__ BinaryDouble(const double& val) {
		assert(sizeof(double) == sizeof(uint64));
		this->bin = this->double_to_uint64(val);
	}

	__DEVICE_HOST__ BinaryDouble(const uint64& val) :
			bin(val) {
	}

	__DEVICE_HOST__ BinaryDouble() {
	}

	__HOST__ uint64 most_significant_bit() {
		assert(sizeof(double) == sizeof(uint64));

		uint64 bit = 0;
		for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
			bit++;
			if (this->bin & i) {
				break;
			}
		}

		return bit;
	}
};

#endif /* BINARYDOUBLE_H_ */
