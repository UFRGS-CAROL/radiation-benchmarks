/*
 * NVMLWrapper.h
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#ifndef NVMLWRAPPER_H_
#define NVMLWRAPPER_H_

#include <nvml.h>
#include <string>
#include <deque>
#include <thread>

namespace rad
{

class NVMLWrapper {
	unsigned device_index;
	nvmlDevice_t device;

	//Multithreading context
	std::thread profiler;

	//NVML EVENT
	nvmlEventSet_t set;

	std::deque<std::string> data_for_iteration;


	static void data_colector(nvmlDevice_t* device, std::deque<std::string>* it_data);

public:
	NVMLWrapper(unsigned device_index);
	virtual ~NVMLWrapper();

	void start_collecting_data();

	void end_collecting_data();

	std::deque<std::string> get_data_from_iteration();

};

}
#endif /* NVMLWRAPPER_H_ */
