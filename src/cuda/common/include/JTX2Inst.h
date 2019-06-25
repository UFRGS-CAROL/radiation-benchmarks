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

namespace rad {
class JTX2Inst {
	//Multithreading context
	std::thread profiler;
	std::string output_log_file;

	std::deque<std::string> data_for_iteration;
	bool thread_running;
	bool collect_data;

	static void data_colector(std::string* output_log_file,
			bool* thread_running, bool* colllect_data);

public:
	JTX2Inst(std::string& output_file);
	virtual ~JTX2Inst();
	void start_profile();
	void end_profile();

//	std::deque<std::string> get_data_from_iteration();
};
}
#endif /* NVMLWRAPPER_H_ */
