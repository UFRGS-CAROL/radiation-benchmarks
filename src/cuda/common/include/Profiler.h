/*
 * Profiler.h
 *
 *  Created on: 04/07/2019
 *      Author: fernando
 */

#ifndef PROFILER_H_
#define PROFILER_H_

#include <thread>	    // std::thread
#include <atomic>		// std::atomic
#include <string>

namespace rad {

#ifndef SLEEP_TIME
#define PROFILER_SLEEP 1000
#else
#define PROFILER_SLEEP SLEEP_TIME
#endif

class Profiler {
protected:
	//Device index of GPU
	unsigned _device_index = 0;

	//Multithreading context
	std::thread _thread_profiler;

	std::atomic<bool> _is_locked;
	std::atomic<bool> _thread_running;

	std::string _output_log_file;

	static void data_colector();

public:
	Profiler(unsigned device_index, std::string& output_file) :
			_device_index(device_index), _is_locked(true),  _thread_running(true), _output_log_file(
					output_file) {
		const std::string to_replace = ".log";
		size_t start_pos = this->_output_log_file.find(to_replace);
		this->_output_log_file.replace(start_pos, to_replace.size(),
				"Profiler.csv");
	}
	virtual ~Profiler() = default;

	virtual void start_profile() = 0;
	virtual void end_profile() = 0;
};

} /* namespace rad */

#endif /* PROFILER_H_ */
