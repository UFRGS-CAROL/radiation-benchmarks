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
	bool allocated_device;
//	/*Will only copy to host if it is requested*/
//	T *host_data;
//	bool allocated_host;

	size_t v_size;

	inline void memcopy(T* dst, T* src, size_t bytes_size);
public:
	DeviceVector(size_t siz);
	DeviceVector();

	//this constructor will copy the data to gpu
	DeviceVector(T *data, size_t siz);

	virtual ~DeviceVector();

	//like std::vector
	void resize(size_t siz);
	T* data();

	DeviceVector<T>& operator=(const std::vector<T>& other);

	//overload only for host side
	T& operator[](int i);

	size_t size();

};

template<class T>
DeviceVector<T>::DeviceVector(size_t siz) {
	cudaMallocManaged(&this->device_data, sizeof(T) * siz);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
	this->v_size = siz;
	this->allocated_device = true;
}

template<class T>
DeviceVector<T>::DeviceVector() {
	this->device_data = nullptr;
	this->v_size = 0;
	this->allocated_device = false;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
	if (this->allocated_device) {
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);

		ret = cudaFree(this->device_data);
		CUDA_CHECK_RETURN(ret);
		this->v_size = 0;
		this->allocated_device = false;
	}
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
	if (this->allocated_device) {
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		cudaFree(this->device_data);
	}
	cudaMallocManaged(&this->device_data, sizeof(T) * siz);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
	this->v_size = siz;
	this->allocated_device = true;
	this->memcopy(this->device_data, data, sizeof(T) * siz);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		if (this->allocated_device) {
			cudaError_t ret = cudaDeviceSynchronize();
			CUDA_CHECK_RETURN(ret);
			cudaFree(this->device_data);
		}
		cudaMallocManaged(&this->device_data, sizeof(T) * siz);
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);

		this->v_size = siz;
		this->allocated_device = true;
		this->memcopy(this->device_data, data, sizeof(T) * siz);

	}
	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
	if (this->v_size != 0) {
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		cudaFree(this->device_data);
	} else if (this->v_size != siz) {

		cudaMallocManaged(&this->device_data, sizeof(T) * siz);
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		this->v_size = siz;
		this->allocated_device = true;
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
T& DeviceVector<T>::operator [](int i){
//	if (!this->allocated_host)
//		this->pop_vector_from_gpu();
//	return this->host_data[i];
	return this->device_data[i];
}

template <class T>
void DeviceVector<T>::memcopy(T* dst, T* src, size_t bytes_size){
	memcpy(dst, src, bytes_size);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}
#endif /* DEVICEVECTOR_H_ */
