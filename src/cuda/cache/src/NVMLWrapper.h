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
#include <thread>
#include <mutex>          // std::mutex
#include <condition_variable>
#include <atomic>

class NVMLWrapper {
private:

	unsigned device_index;
	nvmlDevice_t device;

	//Multithreading context
	std::thread profiler;

	//NVML EVENT
	nvmlEventSet_t set;

	static void data_colector(nvmlDevice_t* device);

public:
	NVMLWrapper(unsigned device_index);
	virtual ~NVMLWrapper();

	void start_collecting_data();

	void end_collecting_data();

};

#endif /* NVMLWRAPPER_H_ */
