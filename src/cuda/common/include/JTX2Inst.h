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
	unsigned device_index;

	//Multithreading context
	std::thread profiler;

	std::deque<std::string> data_for_iteration;

	static void data_colector(std::deque<std::string>* it_data);

public:
	JTX2Inst(unsigned device_index);
	virtual ~JTX2Inst();

	void start_collecting_data();

	void end_collecting_data();

	std::deque<std::string> get_data_from_iteration();
};
}
#endif /* NVMLWRAPPER_H_ */
