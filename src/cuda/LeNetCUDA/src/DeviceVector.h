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
//	bool allocated_device;
//	/*Will only copy to host if it is requested*/
//	T *host_data;
//	bool allocated_host;

	size_t v_size;

	inline void memcopy(T* src, size_t size_count);
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

// Unified memory copy constructor allows pass-by-value
template<class T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& copy){
	int siz = copy.v_size;
	if(this->device_data != nullptr){
		CudaCheckError();
		cudaFree(this->device_data);
	}
	CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
	CudaCheckError();
	this->memcopy(copy.device_data, siz);
}

template<class T>
DeviceVector<T>::DeviceVector(size_t siz) {
	CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
	this->v_size = siz;

}

template<class T>
DeviceVector<T>::DeviceVector() {
	this->device_data = nullptr;
	this->v_size = 0;
}

template<class T>
DeviceVector<T>::~DeviceVector() {
	if (this->device_data != nullptr) {
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		cudaFree(this->device_data);
		CudaCheckError() ;
		this->device_data = nullptr;
		this->v_size = 0;
	}
}

template<class T>
DeviceVector<T>::DeviceVector(T *data, size_t siz) {
	if (this->device_data != nullptr) {
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		cudaFree(this->device_data);
		CudaCheckError();
	}
	CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);

	this->v_size = siz;

	this->memcopy(data, siz);
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const DeviceVector<T>& other) {
	if (this->device_data != other.device_data) { // self-assignment check expected
		T *data = (T*) other.device_data;
		size_t siz = other.v_size;

		if (this->device_data != nullptr) {
			cudaError_t ret = cudaDeviceSynchronize();
			CUDA_CHECK_RETURN(ret);
			cudaFree(this->device_data);
			CudaCheckError();
		}

		CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);

		this->v_size = siz;

		this->memcopy(data, siz);

	}

	return *this;
}

template<class T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other) {
	if (this->data() != other.data()) { // self-assignment check expected
		T *data = (T*) other.data();
		size_t siz = other.size();

		if (this->device_data != nullptr) {
			cudaError_t ret = cudaDeviceSynchronize();
			CUDA_CHECK_RETURN(ret);
			cudaFree(this->device_data);
			CudaCheckError();
		}
		CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);

		this->v_size = siz;
		this->memcopy(data, siz);

	}
	return *this;
}

template<class T>
void DeviceVector<T>::resize(size_t siz) {
	if (this->v_size != siz) {
		if (this->v_size != 0) {
			cudaError_t ret = cudaDeviceSynchronize();
			CUDA_CHECK_RETURN(ret);
			cudaFree(this->device_data);
			CudaCheckError();
		}

		CudaSafeCall(cudaMallocManaged(&this->device_data, sizeof(T) * siz));
		cudaError_t ret = cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(ret);
		this->v_size = siz;
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
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}

template<class T>
void DeviceVector<T>::clear() {
	memset(this->device_data, 0, sizeof(T) * this->v_size);
	cudaError_t ret = cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(ret);
}
#endif /* DEVICEVECTOR_H_ */
