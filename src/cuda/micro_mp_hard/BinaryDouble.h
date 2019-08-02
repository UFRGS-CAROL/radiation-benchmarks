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

	__DEVICE_HOST_ operator double() {
		double val;
		assert(sizeof(double) == sizeof(uint64));
		memcpy(&val, &this->bin, sizeof(double));

		return val;
	}

	__DEVICE_HOST_ BinaryDouble operator&(const BinaryDouble& r) {
		return BinaryDouble(r.bin & this->bin);
	}

	__DEVICE_HOST_ BinaryDouble operator&(const uint64& r) {
		return BinaryDouble(r & this->bin);
	}

	__DEVICE_HOST_ bool operator!=(const uint64& r) {
		return r != this->bin;
	}

	__DEVICE_HOST_ bool operator==(const uint64& r) {
		return r == this->bin;
	}


	__DEVICE_HOST_ BinaryDouble operator^(const BinaryDouble& r) {
		return BinaryDouble(r.bin ^ this->bin);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryDouble& r) {
		assert(sizeof(double) == sizeof(uint64));

		uint64 int_val;

		memcpy(&int_val, &r.bin, sizeof(double));
		for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
			if (int_val & i) {
				os << 1;
			} else {
				os << 0;
			}
		}

		return os;
	}

	__DEVICE_HOST_ BinaryDouble& operator=(const double& val) {
		assert(sizeof(double) == sizeof(uint64));
		memcpy(&this->bin, &val, sizeof(double));

		return *this;
	}

	__DEVICE_HOST_ BinaryDouble(const double& val) {
		assert(sizeof(double) == sizeof(uint64));
		memcpy(&this->bin, &val, sizeof(double));

	}

	__DEVICE_HOST_ BinaryDouble(const uint64& val) :
			bin(val) {
	}

	__DEVICE_HOST_ BinaryDouble() {
	}

	__HOST__ uint64 most_significant_bit() {
		assert(sizeof(double) == sizeof(uint64));
		uint64 int_val;
		memcpy(&int_val, &this->bin, sizeof(double));
		uint64 bit = 0;
		for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
			bit++;
			if (int_val & i) {
				break;
			}
		}

		return bit;
	}
};

#endif /* BINARYDOUBLE_H_ */
