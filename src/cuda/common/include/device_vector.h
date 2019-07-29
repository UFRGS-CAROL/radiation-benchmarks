/*
 * device_vector.h
 *
 *  Created on: 27/03/2019
 *      Author: fernando
 */

#ifndef DEVICE_VECTOR_H_
#define DEVICE_VECTOR_H_

#include <vector>
#include "cuda_utils.h"

namespace rad {

#define VOIDCAST(x) reinterpret_cast<void**>(x)

template<class T>
class DeviceVector {
	T *data_ = nullptr;
	bool allocated = false;
	size_t v_size = 0;

	void alloc_data(size_t size) {
		this->v_size = size;
		checkFrameworkErrors(
				cudaMalloc(VOIDCAST(&this->data_), sizeof(T) * this->v_size));
		this->allocated = true;
	}

	void free_data() {
		if (this->allocated == false)
			return;

		if (this->data_ != nullptr)
			checkFrameworkErrors(cudaFree(this->data_));

		this->allocated = false;
		this->data_ = nullptr;
		this->v_size = 0;
	}

public:
	DeviceVector() {
		this->v_size = 0;
		this->allocated = false;
		this->data_ = nullptr;
	}

	DeviceVector(const DeviceVector<T>& b) {
		this->alloc_data(b.v_size);
		checkFrameworkErrors(
				cudaMemcpy(this->data_, b.data_, sizeof(T) * this->v_size,
						cudaMemcpyDeviceToDevice));
	}

	DeviceVector(const std::vector<T>& b) {
		this->alloc_data(b.size());
		checkFrameworkErrors(
				cudaMemcpy(this->data_, b.data(), sizeof(T) * this->v_size,
						cudaMemcpyHostToDevice));
	}

	DeviceVector(size_t size) {
		this->alloc_data(size);
	}

	virtual ~DeviceVector() {
		this->free_data();
	}

	DeviceVector<T>& operator=(const std::vector<T>& other) {
		if (this->v_size != other.size()) {
			this->free_data();
		}

		if (this->allocated == false) {
			this->alloc_data(other.size());
		}

		checkFrameworkErrors(
				cudaMemcpy(this->data_, other.data(), sizeof(T) * this->v_size,
						cudaMemcpyHostToDevice));

		return *this;
	}

	DeviceVector<T>& operator=(const DeviceVector<T>& other) {
		if (this->v_size != other.v_size) {
			this->free_data();
		}

		if (this->allocated == false) {
			this->alloc_data(other.v_size);

		}

		checkFrameworkErrors(
				cudaMemcpy(this->data_, other.data_, sizeof(T) * this->v_size,
						cudaMemcpyHostToDevice));

		return *this;
	}

	std::vector<T> to_vector() {
		std::vector<T> ret(this->v_size, 0);

		checkFrameworkErrors(
				cudaMemcpy(ret.data(), this->data_, sizeof(T) * this->v_size,
						cudaMemcpyDeviceToHost));
		return ret;
	}

	T* data() {
		return this->data_;
	}

	size_t size() {
		return this->v_size;
	}

	void clear() {
		checkFrameworkErrors(
				cudaMemset(this->data_, 0x0, sizeof(T) * this->v_size));
	}

	void resize(size_t size){
		this->free_data();
		this->alloc_data(size);
	}
};

}

#endif /* DEVICE_VECTOR_H_ */
