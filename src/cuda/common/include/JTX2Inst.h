/*
 * NVMLWrapper.h
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#ifndef NVMLWRAPPER_H_
#define NVMLWRAPPER_H_

#include <string>
#include <deque>
#include <thread>

#include "Profiler.h"


namespace rad {
class JTX2Inst : public Profiler {
	std::deque<std::string> data_for_iteration;
protected:

	static void data_colector(std::string* output_log_file,
			std::atomic<bool>* thread_running, std::atomic<bool>* _is_locked);

public:
	JTX2Inst(unsigned device_index, std::string& output_file);
	virtual ~JTX2Inst();
	void start_profile();
	void end_profile();

//	std::deque<std::string> get_data_from_iteration();
};
}
#endif /* NVMLWRAPPER_H_ */
