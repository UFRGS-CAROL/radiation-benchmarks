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

	BlockList(dim3 old_grid_size){
		// -------------------------------------------------------------------------------------
		cudaDeviceProp prop = GetDevice();

		dim3 dim_grid_full(prop.multiProcessorCount);
		auto grid_size = old_grid_size.x * old_grid_size.y * old_grid_size.z;
		this->block_slice = std::floor(float(grid_size) / prop.multiProcessorCount);
		this->block_slice += grid_size % prop.multiProcessorCount;
		dim3 new_grid_size(gridsize, gridsize, 1);
		// -------------------------------------------------------------------------------------


		std::vector<dim3> temp_vector; //(grid_size.x * grid_size.y * grid_size.z);
		for(auto x = 0; x < new_grid_size.x; x++){
			for(auto y = 0; y < new_grid_size.y; y++){
				for (auto z = 0; z < new_grid_size.z; z++) {
					temp_vector.push_back(dim3(x, y, z));
//					temp_vector[x * grid_size.y * grid_size.z + y * grid_size.z + z] = dim3(x, y, z);
				}
			}
		}
		this->data_ = temp_vector;
	}

	size_t size(){
		return this->data_.size();
	}


	dim3* data(){
		return this->data_;
	}

private:
	rad::DeviceVector<dim3> data_;
	int block_slice;

	cudaDeviceProp GetDevice() {
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

		return prop;
	}

};

#endif /* BLOCKLIST_H_ */
