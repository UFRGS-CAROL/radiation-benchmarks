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
#include <mutex>        // std::mutex

#include "Profiler.h"

namespace rad {

class NVMLWrapper : public Profiler{
	//NVML EVENT
	nvmlEventSet_t _nvml_set;

	nvmlDevice_t _nvml_device;

	std::deque<std::string> data_for_iteration;


	std::mutex _mutex_lock;

protected:

	static void data_colector(nvmlDevice_t* device,
			std::deque<std::string>* it_data, std::mutex* mutex_lock,
			std::atomic<bool>* is_locked, std::atomic<bool>* thread_running);

public:
	NVMLWrapper(unsigned device_index, std::string& output_file);
	virtual ~NVMLWrapper();

	void start_profile();

	void end_profile();

	std::deque<std::string> get_data_from_iteration();

};

}
#endif /* NVMLWRAPPER_H_ */
