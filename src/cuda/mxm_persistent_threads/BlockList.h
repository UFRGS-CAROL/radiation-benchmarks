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
		std::vector<dim3> temp_vector; //(grid_size.x * grid_size.y * grid_size.z);
		for(auto x = 0; x < grid_size.x; x++){
			for(auto y = 0; y < grid_size.y; y++){
				for (auto z = 0; z < grid_size.z; z++) {
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
};

#endif /* BLOCKLIST_H_ */
