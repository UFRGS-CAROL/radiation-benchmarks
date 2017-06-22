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

	void push_vector_to_gpu(T *data);
	void pop_vector_from_gpu();

	T *get_raw_data_host();
	T *get_raw_data_device();

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
	cudaError_t ret = cudaMallocManaged(&this->device_data, sizeof(T) * siz);
	CUDA_CHECK_RETURN(ret);
	this->v_size = siz;
	this->allocated_device = true;
}

template<class T>
DeviceVector<T>::DeviceVector() {
//	this->device_data = this->host_data = nullptr;
	this->device_data = nullptr;
	this->v_size = 0;
	this->allocated_device = false;
//	this->allocated_host = false;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
	if (this->allocated_device) {
		cudaError_t ret = cudaFree(this->device_data);
		CUDA_CHECK_RETURN(ret);
		this->v_size = 0;
		this->allocated_device = false;
	}
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
	if (this->allocated_device) {
		cudaFree(this->device_data);

	}
	cudaError_t ret = cudaMallocManaged(&this->device_data, sizeof(T) * siz);
	CUDA_CHECK_RETURN(ret);
	this->v_size = siz;
	this->allocated_device = true;
	this->memcopy(this->device_data, data, sizeof(T) * siz);
	ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		if (this->allocated_device) {
			cudaFree(this->device_data);
		}
		cudaError_t ret = cudaMallocManaged(&this->device_data, sizeof(T) * siz);
		CUDA_CHECK_RETURN(ret);
		this->v_size = siz;
		this->allocated_device = true;
		this->memcopy(this->device_data, data, sizeof(T) * siz);
		ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
	}
	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
	if (this->v_size != 0) {
		cudaFree(this->device_data);
	} else if (this->v_size != siz) {

		cudaError_t ret = cudaMallocManaged(&this->device_data, sizeof(T) * siz);
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
inline void DeviceVector<T>::pop_vector_from_gpu() {
//	if (this->allocated_host)
//		free(this->host_data);
//
//	this->host_data = (T*) calloc(this->v_size, sizeof(T));
//	cudaError_t ret = cudaMemcpy(this->host_data, this->device_data, sizeof(T) * this->v_size,
//			cudaMemcpyDeviceToHost);
//	CUDA_CHECK_RETURN(ret);
//
//	this->allocated_host = true;
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
}
#endif /* DEVICEVECTOR_H_ */
