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
	// device memory
	T *device_data = nullptr;

	//host memory
	T *host_data = nullptr;

	//to control allocation memory
	bool allocated = false;
	size_t v_size;

	inline void memcopy(T* src, size_t size_count, char src_location = 'd');
	inline void free_memory();
	inline void alloc_memory();

public:
	DeviceVector(size_t siz);
	DeviceVector();
	DeviceVector(const DeviceVector<T>& copy);
	//overload only for host side
	T& operator[](int i) const;
	//this constructor will copy the data to gpu
	DeviceVector(T *data, size_t siz);

	virtual ~DeviceVector();

	//like std::vector
	void resize(size_t siz);
	T* data();
	T* h_data();

	void clear();

	DeviceVector<T>& operator=(const std::vector<T>& other);
	DeviceVector<T>& operator=(const DeviceVector<T>& other);

	void pop();
	void push();
	size_t size() const;

	void fill(T data);

};

template<class T>
inline void DeviceVector<T>::free_memory() {
	CudaCheckError();
	//free device side and host side
	if (this->allocated) {
		CudaSafeCall(cudaFree(this->device_data));
		this->allocated = false;
		this->device_data = nullptr;

		free(this->host_data);
		this->host_data = nullptr;
	}
}

template<class T>
inline void DeviceVector<T>::alloc_memory() {

//#ifdef NOTUNIFIEDMEMORY
//#else
//	CudaSafeCall(
//			cudaMallocManaged(&this->device_data, sizeof(T) * this->v_size));
//#endif
	CudaSafeCall(cudaMalloc(&this->device_data, sizeof(T) * this->v_size));
	this->host_data = (T*) calloc(this->v_size, sizeof(T));
	assert(this->host_data != nullptr && "Error on allocating host data");

	CudaCheckError();
	this->allocated = true;
}

// Unified memory copy constructor allows pass-by-value
template<class T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& copy) {
	this->free_memory();
	this->v_size = copy.v_size;
	this->alloc_memory();
	this->memcopy(copy.device_data, this->v_size);
}

template<class T>
DeviceVector<T>::DeviceVector(size_t siz) {
	this->free_memory();
	this->v_size = siz;
	this->alloc_memory();
}

template<class T>
DeviceVector<T>::DeviceVector() {
	this->device_data = nullptr;
	this->host_data = nullptr;
	this->v_size = 0;
	this->allocated = false;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
	this->free_memory();
	CudaCheckError();
	this->v_size = 0;
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
	this->free_memory();
	this->v_size = siz;
	this->alloc_memory();
	this->memcopy(data, siz);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const DeviceVector<T>& other) {
	if (this->device_data != other.device_data) { // self-assignment check expected
		T *data = (T*) other.device_data;

		this->free_memory();

		this->v_size = other.v_size;
		this->alloc_memory();
		this->memcopy(data, this->v_size);
	}

	return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		this->free_memory();
		this->v_size = siz;
		this->alloc_memory();
		this->memcopy(data, siz, 'h');
	}

	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
	if (this->v_size != siz) {
		this->free_memory();
		this->v_size = siz;
		this->alloc_memory();
	}
}

template<class T>
T* DeviceVector<T>::data() {
	return this->device_data;
}

template<class T>
size_t DeviceVector<T>::size() const {
	return this->v_size;
}

template<class T>
T& DeviceVector<T>::operator [](int i) const {
//#ifdef NOTUNIFIEDMEMORY//#else
//	return this->device_data[i];
//#endif
	return this->host_data[i];
}

template<class T>
void DeviceVector<T>::memcopy(T* src, size_t size_cont, char src_location) {
//#ifdef NOTUNIFIEDMEMORY
//#else
//	memcpy(this->device_data, src, sizeof(T) * size_cont);
//#endif
	if (src_location == 'd') {
		CudaSafeCall(
				cudaMemcpy(this->device_data, src, sizeof(T) * size_cont,
						cudaMemcpyDeviceToDevice));
	} else {
		CudaSafeCall(
				cudaMemcpy(this->device_data, src, sizeof(T) * size_cont,
						cudaMemcpyHostToDevice));
	}
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::clear() {
//#ifdef NOTUNIFIEDMEMORY
//#else
//	memset(this->device_data, 0, sizeof(T) * this->v_size);
//#endif
	CudaSafeCall(cudaMemset(this->device_data, 0, sizeof(T) * this->v_size));
	memset(this->host_data, 0, sizeof(T) * this->v_size);
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::fill(T data) {
//#ifdef NOTUNIFIEDMEMORY
//#else
//	memset(this->device_data, data, sizeof(T) * this->v_size);
//#endif
	std::fill_n(this->host_data, this->v_size, data);
	this->push();
}

template<class T>
void DeviceVector<T>::pop() {
	CudaSafeCall(
			cudaMemcpy(this->host_data, this->device_data,
					sizeof(T) * this->v_size, cudaMemcpyDeviceToHost));
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::push() {
	CudaSafeCall(
			cudaMemcpy(this->device_data, this->host_data,
					sizeof(T) * this->v_size, cudaMemcpyHostToDevice));
	CudaCheckError();
}

template<class T>
T* DeviceVector<T>::h_data() {
	return this->host_data;
}

#endif /* DEVICEVECTOR_H_ */
