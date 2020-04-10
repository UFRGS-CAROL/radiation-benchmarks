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
	DeviceVector() :
			v_size(0), allocated(false), data_(nullptr) {
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

	DeviceVector(size_t size, const T val) {
		std::vector<T> temp_array(size, 0);
		this->alloc_data(size);
		checkFrameworkErrors(
				cudaMemcpy(this->data_, temp_array.data(), sizeof(T) * this->v_size,
						cudaMemcpyHostToDevice));
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
						cudaMemcpyDeviceToDevice));

		return *this;
	}

	std::vector<T> to_vector() {
		std::vector < T > ret(this->v_size);

		checkFrameworkErrors(
				cudaMemcpy(ret.data(), this->data_, sizeof(T) * this->v_size,
						cudaMemcpyDeviceToHost));
		return ret;
	}

	void to_vector(std::vector<T>& lhs) {
		if (lhs.size() != this->v_size) {
			lhs.resize(this->v_size);
		}

		checkFrameworkErrors(
				cudaMemcpy(lhs.data(), this->data_, sizeof(T) * this->v_size,
						cudaMemcpyDeviceToHost));
	}

	void to_vector_async(std::vector<T>& lhs, const cudaStream_t& stream) {
		if (lhs.size() != this->v_size) {
			lhs.resize(this->v_size);
		}

		checkFrameworkErrors(
				cudaMemcpyAsync(lhs.data(), this->data_,
						sizeof(T) * this->v_size, cudaMemcpyDeviceToHost,
						stream));
	}

	T* data() const {
		return this->data_;
	}

	size_t size() {
		return this->v_size;
	}

	void clear() {
		checkFrameworkErrors(
				cudaMemset(this->data_, 0x0, sizeof(T) * this->v_size));
	}

	void resize(size_t size) {
		this->free_data();
		this->alloc_data(size);
	}

	template<typename Iterator>
	void fill_n(Iterator begin, size_t n) {
		if ((this->data_ + n) <= (this->data_ + this->v_size)) {
			checkFrameworkErrors(
					cudaMemcpy(this->data_, &(*(begin)), sizeof(T) * n,
							cudaMemcpyHostToDevice));
		}
	}

	template<typename Iterator>
	void get_n(Iterator begin, size_t n) {
		if ((this->data_ + n) <= (this->data_ + this->v_size)) {
			checkFrameworkErrors(
					cudaMemcpy(&(*(begin)), this->data_, sizeof(T) * n,
							cudaMemcpyDeviceToHost));
		}
	}
};

}

#endif /* DEVICE_VECTOR_H_ */
