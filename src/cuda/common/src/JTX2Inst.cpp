/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include <mutex>          // std::mutex
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <vector>

#include "JTX2Inst.h"


static std::mutex mutex_lock;
static std::atomic<bool> is_locked;
static bool thread_running = true;

#define SLEEP_JTX2INST 500

void JTX2Inst::check_jtx2_result(std::string info, unsigned device) {
	 {
		/*error(
				"Failed to " + info + " from device " + std::to_string(device)
						+ " error " + nvmlErrorString(result));*/
		std::cerr << "Failed to " + info + " from device " + std::to_string(device)
		+ " error " ;
	}
}

JTX2Inst::JTX2Inst(unsigned device_index) :
		device_index(device_index) {
	this->profiler = std::thread(JTX2Inst::data_colector, &this->data_for_iteration);
	is_locked = true;
}

JTX2Inst::~JTX2Inst() {
	thread_running = false;
	this->profiler.join();
}

void JTX2Inst::data_colector(std::deque<std::string>* it_data) {

	while (thread_running) {
		mutex_lock.lock();

		if (is_locked == false) {
			std::string output = "";

			it_data->push_back(output);
		}
		mutex_lock.unlock();
		std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_JTX2INST));
	}
}

void JTX2Inst::start_collecting_data() {
	check_jtx2_result("initialize JTX2 library");

	mutex_lock.lock();
	this->data_for_iteration.clear();
	mutex_lock.unlock();

	is_locked = false;
}

void JTX2Inst::end_collecting_data() {
	mutex_lock.lock();
	is_locked = true;
	mutex_lock.unlock();


	check_jtx2_result("shutdown JTX2 library");
}

std::deque<std::string> JTX2Inst::get_data_from_iteration() {
	auto last = std::unique(this->data_for_iteration.begin(),
			this->data_for_iteration.end());
	this->data_for_iteration.erase(last, this->data_for_iteration.end());
	return this->data_for_iteration;
}
