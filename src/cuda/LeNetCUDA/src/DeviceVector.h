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
	/*Will only copy to host if it is requested*/
	T *host_data;
	bool allocated_host;

	size_t size;

public:
	DeviceVector(size_t siz);

	virtual ~DeviceVector();

	void push_vector_to_gpu(T *data);
	void pop_vector_from_gpu();

	T *get_raw_data_host();
	T *get_raw_data_device();

};

template<class T> DeviceVector<T>::DeviceVector(size_t siz) {
	cudaError_t ret = cudaMalloc(&this->device_data, sizeof(T) * siz);
	CUDA_CHECK_RETURN(ret);
	this->size = siz;
	this->allocated_device = true;
}

template<class T> DeviceVector<T>::~DeviceVector() {
	if (this->allocated_device) {
		cudaError_t ret = cudaFree(this->device_data)
		CUDA_CHECK_RETURN(ret);
		this->size = 0;
		this->allocated_device = false;
	}
}

#endif /* DEVICEVECTOR_H_ */
