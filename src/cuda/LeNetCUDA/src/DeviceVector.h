/*
 * DeviceVector.h
 *
 *  Created on: 14/06/2017
 *      Author: fernando
 */

#ifndef DEVICEVECTOR_H_
#define DEVICEVECTOR_H_

#include <vector>
#include "cudaUtil.h"

/**
 * Template for vector on the GPU
 */
template<class T> class DeviceVector {
private:
	T *device_data;
	bool allocated;

	size_t v_size;

	inline void memcopy(T* src, size_t size_count);
//	inline void free_memory();
//	inline void alloc_memory();

public:
	DeviceVector(size_t siz);
	DeviceVector();
	DeviceVector(const DeviceVector<T>& copy);

	//this constructor will copy the data to gpu
	DeviceVector(T *data, size_t siz);

	virtual ~DeviceVector();

	//like std::vector
	void resize(size_t siz);
	T* data();

	void clear();

	DeviceVector<T>& operator=(const std::vector<T>& other);
	DeviceVector<T>& operator=(const DeviceVector<T>& other);

	//overload only for host side
	T& operator[](int i);

	size_t size();

};

template<class T>
//inline void DeviceVector<T>::free_memory() {
inline void free_memory(T* device_data, bool *allocated){
	CudaCheckError();
	CudaSafeCall(cudaFree(device_data));
	*allocated = false;
}

template<class T>
//inline void DeviceVector<T>::alloc_memory() {
inline void alloc_memory(T *device_data, size_t v_size, bool *allocated){
	CudaSafeCall(
			cudaMallocManaged(&device_data, sizeof(T) * v_size));
	CudaCheckError();
	*allocated = true;
}
// Unified memory copy constructor allows pass-by-value
template<class T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& copy) {
	if (this->allocated) {
		free_memory<T>(this->device_data, &this->allocated);
	}

	this->v_size = copy.v_size;
	alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
	this->memcopy(copy.device_data, this->v_size);
}

template<class T>
DeviceVector<T>::DeviceVector(size_t siz) {
	if (this->allocated) {
		free_memory<T>(this->device_data, &this->allocated);
	}
	this->v_size = siz;
	alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
}

template<class T>
DeviceVector<T>::DeviceVector() {
	this->device_data = nullptr;
	this->v_size = 0;
	this->allocated = false;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
	if (this->allocated) {
		free_memory<T>(this->device_data, &this->allocated);
		CudaCheckError();
		this->device_data = nullptr;
		this->v_size = 0;
		this->allocated = false;
	}
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
	if (this->allocated) {
		free_memory<T>(this->device_data, &this->allocated);
	}

	this->v_size = siz;
	alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
	this->memcopy(data, siz);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const DeviceVector<T>& other) {
	if (this->device_data != other.device_data) { // self-assignment check expected
		T *data = (T*) other.device_data;

		if (this->allocated) {
			free_memory<T>(this->device_data, &this->allocated);
		}

		this->v_size = other.v_size;
		alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
		this->memcopy(data, other.v_size);
	}

	return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		if (this->allocated) {
			free_memory<T>(this->device_data, &this->allocated);
		}

		this->v_size = siz;
		alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
		this->memcopy(data, siz);
	}
	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
	if (this->v_size != siz) {
		if (this->v_size != 0) {
			free_memory<T>(this->device_data, &this->allocated);
		}
		this->v_size = siz;
		alloc_memory<T>(this->device_data, this->v_size, &this->allocated);
		this->allocated = true;
	}
}

template<class T>
T* DeviceVector<T>::data() {
	return this->device_data;
}

template<class T>
size_t DeviceVector<T>::size() {
	return this->v_size;
}

template<class T>
T& DeviceVector<T>::operator [](int i) {
	return this->device_data[i];
}

template<class T>
void DeviceVector<T>::memcopy(T* src, size_t size_cont) {
	memcpy(this->device_data, src, sizeof(T) * size_cont);
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::clear() {
	memset(this->device_data, 0, sizeof(T) * this->v_size);
	CudaCheckError();
}
#endif /* DEVICEVECTOR_H_ */
