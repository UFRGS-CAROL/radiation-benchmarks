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

	__DEVICE_HOST__ uint64 double_to_uint64(const double d) {
		const double* ptr = &d;
		const uint64* ptr_i = (const uint64*) ptr;
		return *ptr_i;
	}

	__DEVICE_HOST__ double uint64_to_double(const uint64 u) {
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
		this->bin = this->double_to_uint64(val);
		return *this;
	}

	__DEVICE_HOST__ BinaryDouble(const double& val) {
		assert(sizeof(double) == sizeof(uint64));
		this->bin = this->double_to_uint64(val);
	}

	__DEVICE_HOST__ BinaryDouble(const uint64& val) :
			bin(val) {
		assert(sizeof(double) == sizeof(uint64));

	}

	__DEVICE_HOST__ BinaryDouble() {
		assert(sizeof(double) == sizeof(uint64));
	}

	__DEVICE_HOST__ uint64 most_significant_bit() const {
		uint64 bit = 0;
		for (uint64 i = uint64(1) << 63; i > 0; i = i / 2) {
			bit++;
			if (this->bin & i) {
				break;
			}
		}

		return bit;
	}

	bool operator<(const BinaryDouble& rhs) const {
		auto key_this = this->most_significant_bit();
		auto key_rhs = rhs.most_significant_bit();
		return (key_this < key_rhs);
	}

	bool operator>(const BinaryDouble& rhs) const {
		auto key_this = this->most_significant_bit();
		auto key_rhs = rhs.most_significant_bit();
		return (key_this > key_rhs);
	}
};

#define SHFT_FLOAT 31
struct BinaryFloat {
	uint32 bin;

	__DEVICE_HOST__ uint32 float_to_uint32(const float d) {
		const float* ptr = &d;
		const uint32* ptr_i = (const uint32*) ptr;
		return *ptr_i;
	}

	__DEVICE_HOST__ float uint32_to_float(const uint32 u) {
		const uint32* ptr = &u;
		const float* ptr_d = (const float*) ptr;
		return *ptr_d;
	}

	__DEVICE_HOST__ operator float() {
		return this->uint32_to_float(this->bin);
	}

	__DEVICE_HOST__ BinaryFloat operator&(const BinaryFloat& r) {
		return BinaryFloat(r.bin & this->bin);
	}

	__DEVICE_HOST__ BinaryFloat operator&(const uint32& r) {
		return BinaryFloat(r & this->bin);
	}

	__DEVICE_HOST__ BinaryFloat operator-(const BinaryFloat& r){
		return BinaryFloat(r.bin > this->bin ? r.bin - this->bin : this->bin - r.bin);
	}

	__DEVICE_HOST__ bool operator!=(const uint32& r) {
		return r != this->bin;
	}

	__DEVICE_HOST__ bool operator==(const uint32& r) {
		return r == this->bin;
	}

	__DEVICE_HOST__ BinaryFloat operator^(const BinaryFloat& r) {
		return BinaryFloat(r.bin ^ this->bin);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryFloat& r) {
		for (uint32 i = uint32(1) << SHFT_FLOAT; i > 0; i = i / 2) {
			if (r.bin & i) {
				os << 1;
			} else {
				os << 0;
			}
		}

		return os;
	}

	__DEVICE_HOST__ BinaryFloat& operator=(const float& val) {
		this->bin = this->float_to_uint32(val);
		return *this;
	}

	__DEVICE_HOST__ BinaryFloat(const float& val) {
		assert(sizeof(float) == sizeof(uint32));
		this->bin = this->float_to_uint32(val);
	}

	__DEVICE_HOST__ BinaryFloat(const uint32& val) :
			bin(val) {
		assert(sizeof(float) == sizeof(uint32));

	}

	__DEVICE_HOST__ BinaryFloat() {
		assert(sizeof(float) == sizeof(uint32));
	}

	__DEVICE_HOST__ uint32 most_significant_bit() const {
		uint32 bit = 0;
		for (uint32 i = uint32(1) << SHFT_FLOAT; i > 0; i = i / 2) {
			bit++;
			if (this->bin & i) {
				break;
			}
		}

		return bit;
	}

	bool operator<(const BinaryFloat& rhs) const {
		auto key_this = this->most_significant_bit();
		auto key_rhs = rhs.most_significant_bit();
		return (key_this < key_rhs);
	}

	bool operator>(const BinaryFloat& rhs) const {
		auto key_this = this->most_significant_bit();
		auto key_rhs = rhs.most_significant_bit();
		return (key_this > key_rhs);
	}
};

#endif /* BINARYDOUBLE_H_ */
