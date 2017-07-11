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
//#define DEBUG_LIGHT 

/**
 * Template for vector on the GPU
 */
template<class T> class DeviceVector {
private:
	T *device_data = nullptr;
	bool allocated = false;

	size_t v_size;

	inline void memcopy(T* src, size_t size_count);
	inline void free_memory();
	inline void alloc_memory();

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

	void fill(T data);

};

template<class T>
inline void DeviceVector<T>::free_memory() {
	CudaCheckError();
	CudaSafeCall(cudaFree(this->device_data));
	this->allocated = false;
}

template<class T>
inline void DeviceVector<T>::alloc_memory() {
	CudaSafeCall(
			cudaMallocManaged(&this->device_data, sizeof(T) * this->v_size));
	CudaCheckError();
	this->allocated = true;
}
// Unified memory copy constructor allows pass-by-value
template<class T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& copy) {
#ifdef DEBUG_LIGHT
	std::cout << "DeviceVector(const DeviceVector<T>& copy)\n";
#endif
	if (this->allocated) {
		this->free_memory();
	}

	this->v_size = copy.v_size;
	this->alloc_memory();
	this->memcopy(copy.device_data, this->v_size);
}

template<class T>
DeviceVector<T>::DeviceVector(size_t siz) {
#ifdef DEBUG_LIGHT
	std::cout << "DeviceVector(size_t siz)\n";
#endif
	if (this->allocated) {
		this->free_memory();
	}
	this->v_size = siz;
	this->alloc_memory();
}

template<class T>
DeviceVector<T>::DeviceVector() {
#ifdef DEBUG_LIGHT
	std::cout << "DeviceVector()\n";
#endif
	this->device_data = nullptr;
	this->v_size = 0;
	this->allocated = false;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
#ifdef DEBUG_LIGHT
	std::cout << "~DeviceVector()\n";
#endif
	if (this->allocated) {
		this->free_memory();
		CudaCheckError();
		this->device_data = nullptr;
		this->v_size = 0;
		this->allocated = false;
	}
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
#ifdef DEBUG_LIGHT
	std::cout << "DeviceVector(T *data, size_t siz)\n";
#endif
	if (this->allocated) {
		this->free_memory();
	}

	this->v_size = siz;
	this->alloc_memory();
	this->memcopy(data, siz);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const DeviceVector<T>& other) {
#ifdef DEBUG_LIGHT
	std::cout << "operator=(const DeviceVector<T>& other)\n";
#endif
	if (this->device_data != other.device_data) { // self-assignment check expected
		T *data = (T*) other.device_data;

		if (this->allocated) {
			this->free_memory();
		}

		this->v_size = other.v_size;
		this->alloc_memory();
		this->memcopy(data, this->v_size);
	}

	return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
#ifdef DEBUG_LIGHT
	std::cout << "operator=(const std::vector<T>& other) \n";
#endif
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		if (this->allocated) {
			this->free_memory();
		}

		this->v_size = siz;
		this->alloc_memory();
		this->memcopy(data, siz);
	}
	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
#ifdef DEBUG_LIGHT
	std::cout << "resize(size_t siz)\n";
#endif
	if (this->v_size != siz) {
		if (this->v_size != 0) {
			this->free_memory();
		}
		this->v_size = siz;
		this->alloc_memory();
		this->allocated = true;
	}
}

template<class T>
T* DeviceVector<T>::data() {
#ifdef DEBUG_LIGHT
	std::cout << "data() \n";
#endif
	return this->device_data;
}

template<class T>
size_t DeviceVector<T>::size() {
#ifdef DEBUG_LIGHT
	std::cout << "size() \n";
#endif
	return this->v_size;
}

template<class T>
T& DeviceVector<T>::operator [](int i) {
#ifdef DEBUG_LIGHT
	std::cout << "operator [] \n";
#endif
	return this->device_data[i];
}

template<class T>
void DeviceVector<T>::memcopy(T* src, size_t size_cont) {
#ifdef DEBUG_LIGHT
	std::cout << "memcopy(T* src, size_t size_cont)\n";
#endif
	memcpy(this->device_data, src, sizeof(T) * size_cont);
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::clear() {
#ifdef DEBUG_LIGHT
	std::cout << "clear()\n";
#endif
	memset(this->device_data, 0, sizeof(T) * this->v_size);
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::fill(T data) {
	memset(this->device_data, data, sizeof(T) * this->v_size);
	CudaCheckError();
}
#endif /* DEVICEVECTOR_H_ */
