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

#ifdef NOTUNIFIEDMEMORY
	T *host_data = nullptr;
	bool host_allocated = false;
#endif

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
#ifdef NOTUNIFIEDMEMORY
	T* h_data();
#endif

	void clear();

	DeviceVector<T>& operator=(const std::vector<T>& other);
	DeviceVector<T>& operator=(const DeviceVector<T>& other);

#ifdef NOTUNIFIEDMEMORY
	void pop_vector();
	void push_vector();
#endif
	size_t size() const;

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

#ifdef NOTUNIFIEDMEMORY
	CudaSafeCall(
			cudaMalloc(&this->device_data, sizeof(T) * this->v_size));

#else
	CudaSafeCall(
			cudaMallocManaged(&this->device_data, sizeof(T) * this->v_size));
#endif
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

#ifdef NOTUNIFIEDMEMORY
	if (this->host_allocated){
		free(this->host_data);
		this->host_allocated = false;
		this->host_data = nullptr;
	}
#endif
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
		this->memcopy(data, siz, 'h');
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

#ifdef NOTUNIFIEDMEMORY
		if(this->host_data){
			free(this->host_data);
		}
		this->host_data = (T*) calloc(this->v_size, sizeof(T));
#endif
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
size_t DeviceVector<T>::size() const {
#ifdef DEBUG_LIGHT
	std::cout << "size() \n";
#endif
	return this->v_size;
}


template<class T>
T& DeviceVector<T>::operator [](int i) const {
#ifdef DEBUG_LIGHT
	std::cout << "operator [] \n";
#endif

#ifdef NOTUNIFIEDMEMORY
	return this->host_data[i];
#else
	return this->device_data[i];
#endif
}


template<class T>
void DeviceVector<T>::memcopy(T* src, size_t size_cont, char src_location) {
#ifdef DEBUG_LIGHT
	std::cout << "memcopy(T* src, size_t size_cont)\n";
#endif

#ifdef NOTUNIFIEDMEMORY
	if (src_location == 'd') {
		CudaSafeCall(cudaMemcpy(this->device_data, src, sizeof(T) * size_cont, cudaMemcpyDeviceToDevice));
	} else {
		CudaSafeCall(cudaMemcpy(this->device_data, src, sizeof(T) * size_cont, cudaMemcpyHostToDevice));
	}
#else
	memcpy(this->device_data, src, sizeof(T) * size_cont);
#endif
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::clear() {
#ifdef DEBUG_LIGHT
	std::cout << "clear()\n";
#endif

#ifdef NOTUNIFIEDMEMORY
	CudaSafeCall(cudaMemset(this->device_data, 0, sizeof(T) * this->v_size));
#else
	memset(this->device_data, 0, sizeof(T) * this->v_size);
#endif
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::fill(T data) {
#ifdef NOTUNIFIEDMEMORY
	CudaSafeCall(cudaMemset(this->device_data, data, sizeof(T) * this->v_size));
#else
	memset(this->device_data, data, sizeof(T) * this->v_size);
#endif
	CudaCheckError();
}

#ifdef NOTUNIFIEDMEMORY
template<class T>
void DeviceVector<T>::pop_vector(){
	if (this->host_allocated == false){
		this->host_data = (T*) calloc(this->v_size, sizeof(T));
		this->host_allocated = true;
	}

	CudaSafeCall(cudaMemcpy(this->host_data, this->device_data, sizeof(T) * this->v_size, cudaMemcpyDeviceToHost));
	CudaCheckError();
}

template<class T>
void DeviceVector<T>::push_vector(){
	CudaSafeCall(cudaMemcpy(this->device_data, this->host_data, sizeof(T) * this->v_size, cudaMemcpyHostToDevice));
	CudaCheckError();
}

template<class T>
T* DeviceVector<T>::h_data(){
	return this->host_data;
}

#endif /* NOTUNIFIEDMEMORY */

#endif /* DEVICEVECTOR_H_ */
