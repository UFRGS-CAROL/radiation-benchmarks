/*
 * Log.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef LOG_H_
#define LOG_H_

#ifdef LOGS
#include "log_helper.h"

#ifdef BUILDPROFILER
#include <memory>
#include "NVMLWrapper.h"

#endif //NVML profiler

#endif //log helper

#include <string>
#include <iostream>

namespace rad {

struct Log {
	uint64_t error;
	uint64_t info;
	std::string test_name;
	std::string test_info;

	bool was_error_updated;
	bool was_info_updated;

#ifdef LOGS
#ifdef BUILDPROFILER
	std::shared_ptr<Profiler> profiler_thread;
#endif
#endif

	friend std::ostream& operator<<(std::ostream& os, Log& d) {
		std::string file_name = "No log file name, build with the libraries";
#ifdef LOGS
		file_name = get_log_file_name();
#endif
		os << "LOGFILENAME: " << file_name << std::endl;
		os << "Error: " << d.error << std::endl;
		os << "Info: " << d.info << std::endl;
		os << "Test info: " << d.test_info << std::endl;
		os << "Test name: " << d.test_name;
		return os;
	}

	Log() :
			error(0), info(0), was_error_updated(false), was_info_updated(false) {
	}

	Log(std::string test_name, std::string test_info, size_t print_interval = 1) :
			error(0), info(0), test_name(test_name), test_info(test_info), was_error_updated(
					false), was_info_updated(false) {
#ifdef LOGS
		start_log_file(const_cast<char*>(test_name.c_str()),
				const_cast<char*>(test_info.c_str()));

		::set_iter_interval_print(print_interval);
		/**
		 * Profiler macros to define profiler thread
		 * must build libraries before build with this macro
		 */
#ifdef BUILDPROFILER
		std::string log_file_name(get_log_file_name());
		this->profiler_thread = std::make_shared<NVMLWrapper>(0, log_file_name);

		//START PROFILER THREAD
		profiler_thread->start_profile();
#endif
#endif
	}

	~Log() {
		if (this->was_error_updated == false)
			this->update_errors();
		if (this->was_info_updated == false)
			this->update_infos();

#ifdef LOGS
		::end_log_file();

#ifdef BUILDPROFILER
		this->profiler_thread->end_profile();
#endif

#endif
	}

	void log_error_detail(std::string error_detail) {
		this->error++;
		this->was_error_updated = false;
#ifdef LOGS
		::log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void log_info_detail(std::string info_detail) {
		this->info++;
		this->was_info_updated = false;
#ifdef LOGS
		::log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	void start_iteration() {
		if (this->was_error_updated == false)
			this->update_errors();
		if (this->was_info_updated == false)
			this->update_infos();

		this->error = 0;
		this->info = 0;
#ifdef LOGS
		::start_iteration();
#endif
	}

	void end_iteration() {
#ifdef LOGS
		::end_iteration();
#endif
	}

	void update_errors() {
		this->was_error_updated = true;
		if (this->error != 0) {
#ifdef LOGS
			::log_error_count(this->error);
#endif
		}
	}

	void update_infos() {
		this->was_info_updated = true;
		if (this->info != 0) {
#ifdef LOGS
			::log_info_count(this->info);
#endif
		}
	}

	void set_iter_interval_print(size_t print_interval) {
#ifdef LOGS
		::set_iter_interval_print(print_interval);
#endif
	}

private:
	//Hide copy constructor to avoid copies
	//pass only as reference to the function/method
	Log(const Log& l) :
			error(l.error), info(l.info), was_error_updated(
					l.was_error_updated), was_info_updated(l.was_info_updated) {
	}

};

}

#endif /* LOG_H_ */
