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
//#include <deque>
#include <mutex>        // std::mutex

#include "Profiler.h"

namespace rad {

class NVMLWrapper: public Profiler {
	//NVML EVENT
	nvmlEventSet_t _nvml_set;

	nvmlDevice_t _nvml_device;

//	std::deque<std::string> data_for_iteration;

	std::mutex _mutex_lock;

	bool _persistent_threads;
protected:

	static void data_colector(nvmlDevice_t* device, std::mutex* mutex_lock,
			std::atomic<bool>* is_locked, std::atomic<bool>* thread_running,
			std::string* output_log_file, bool persistent_threads);

public:
	NVMLWrapper(unsigned device_index, std::string& output_file);
	virtual ~NVMLWrapper();

	void start_profile();

	void end_profile();

//	std::deque<std::string> get_data_from_iteration();

	static std::string generate_line_info(nvmlDevice_t* device, bool persistent_threads);

};

}
#endif /* NVMLWRAPPER_H_ */
