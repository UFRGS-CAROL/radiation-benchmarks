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

	BlockList(dim3 grid_size){
		std::vector<dim3> temp_vector(grid_size.x * grid_size.y * grid_size.z);
		for(auto i = 0; i < grid_size.x; i++){
			for(auto j = 0; j < grid_size.y; j++){
				for (auto k = 0; k < grid_size.z; k++) {
					temp_vector[i * grid_size.y * grid_size.z + j * grid_size.z + k] = dim3(i, j, k);
				}
			}
		}
		this->data_ = temp_vector;
	}

	size_t size(){
		return this->data_.size();
	}
};

#endif /* BLOCKLIST_H_ */
