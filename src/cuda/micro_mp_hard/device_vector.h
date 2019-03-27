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

template<class T>
class DeviceVector{
private:
	T *device_data = nullptr;

	bool allocated = false;
	size_t v_size;

public:

	DeviceVector(size_t size){
		this->v_size = size;
		checkFrameworkErrors(cudaMalloc(&this->device_data, sizeof(T) * this->v_size));
		this->allocated = true;
	}

	virtual ~DeviceVector(){
		if(this->allocated){
			checkFrameworkErrors(cudaFree(this->device_data));
		}

		this->allocated = false;
	}

	T& operator [](int i) const {
		return this->host_data[i];
	}

	DeviceVector<T>& operator=(const std::vector<T>& other) {
		if(this->v_size != other.size()){
			checkFrameworkErrors(cudaFree(this->device_data));
			this->allocated = false;
		}

		if (this->allocated == false) {
			this->v_size = other.size();
			checkFrameworkErrors(cudaMalloc(&this->device_data, sizeof(T) * this->v_size));
			this->allocated = true;
		}


		checkFrameworkErrors(cudaMemcpy(this->device_data, other.data(), sizeof(T) * this->v_size,
								cudaMemcpyHostToDevice));

		return *this;
	}


	std::vector<T> operator=(const DeviceVector<T>& other){
		std::vector<T> ret(other.v_size);

		checkFrameworkErrors(cudaMemcpy(ret.data(), other.device_data, sizeof(T) * this->v_size, cudaMemcpyDeviceToHost));
		return ret;
	}

};


#endif /* DEVICE_VECTOR_H_ */
