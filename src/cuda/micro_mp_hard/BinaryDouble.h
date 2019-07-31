/*
 * BinaryDouble.h
 *
 *  Created on: Jul 30, 2019
 *      Author: fernando
 */

#ifndef BINARYDOUBLE_H_
#define BINARYDOUBLE_H_

#define __DEVICE_HOST_ __device__ __host__

struct BinaryDouble {
	uint64_t bin;

	__DEVICE_HOST_ operator double() {
		double val;
		memcpy(&val, &this->bin, sizeof(double));
		return val;
	}

	__DEVICE_HOST_ BinaryDouble operator&(const BinaryDouble& r) {
		return BinaryDouble(r.bin & this->bin);
	}

	__DEVICE_HOST_ BinaryDouble operator^(const BinaryDouble& r) {
		return BinaryDouble(r.bin ^ this->bin);
	}

	friend std::ostream& operator<<(std::ostream& os, const BinaryDouble& r) {
		if (sizeof(double) == 8) {

			uint64_t int_val;

			memcpy(&int_val, &r.bin, sizeof(double));
			for (uint64_t i = uint64_t(1) << 63; i > 0; i = i / 2) {
				if (int_val & i) {
					os << 1;
				} else {
					os << 0;
				}
			}
		}
		return os;
	}

	__DEVICE_HOST_ BinaryDouble& operator=(const double& val) {
		if (sizeof(double) == 8) {
			memcpy(&this->bin, &val, sizeof(double));
		}
		return *this;
	}

	__DEVICE_HOST_ BinaryDouble(const double& val) {
		if (sizeof(double) == 8) {
			memcpy(&this->bin, &val, sizeof(double));
		}
	}

	__DEVICE_HOST_ BinaryDouble(const uint64_t& val) :
			bin(val) {
	}

	__DEVICE_HOST_ BinaryDouble() {
	}
};

#endif /* BINARYDOUBLE_H_ */
