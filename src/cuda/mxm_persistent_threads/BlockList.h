/*
 * BlockList.h
 *
 *  Created on: Aug 29, 2019
 *      Author: carol
 */

#ifndef BLOCKLIST_H_
#define BLOCKLIST_H_

#include "device_vector.h"
#include <vector>

struct BlockList{
	rad::DeviceVector<dim3> data_;
	int block_slice;
	int sm_count;


	BlockList(dim3 old_grid_size) : sm_count(1){
		// -------------------------------------------------------------------------------------
		 this->get_device();

    	auto grid_size = old_grid_size.x * old_grid_size.y * old_grid_size.z;
		this->block_slice = std::ceil(float(grid_size) / this->sm_count);

		// -------------------------------------------------------------------------------------
		std::vector<dim3> temp_vector;
		for(auto x = 0; x < old_grid_size.x; x++){
			for(auto y = 0; y < old_grid_size.y; y++){
				for (auto z = 0; z < old_grid_size.z; z++) {
					temp_vector.push_back(dim3(x, y, z));
				}
			}
		}
		this->data_ = temp_vector;
	}

	size_t size(){
		return this->data_.size();
	}


	dim3* data(){
		return this->data_.data();
	}

	dim3 sm_count_to_dim3(){
		return dim3(this->sm_count, 1, 1);
	}

	cudaDeviceProp get_device() {
	//================== Retrieve and set the default CUDA device
		cudaDeviceProp prop;
		int count = 0;

		rad::checkFrameworkErrors(cudaGetDeviceCount(&count));
		for (int i = 0; i < count; i++) {
			rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, i));
		}
		int *ndevice;
		int dev = 0;
		ndevice = &dev;
		rad::checkFrameworkErrors(cudaGetDevice(ndevice));

		rad::checkFrameworkErrors(cudaSetDevice(0));
		rad::checkFrameworkErrors(cudaGetDeviceProperties(&prop, 0));

		this->sm_count = prop.multiProcessorCount;
		return prop;
	}

};

#endif /* BLOCKLIST_H_ */
