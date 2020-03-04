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
	Log() :
			_error(0), _info(0), _was_error_updated(false), _was_info_updated(
					false) {
	}

	Log(std::string test_name, std::string test_info, size_t print_interval = 1) :
			_error(0), _info(0), _test_name(test_name), _test_info(test_info), _was_error_updated(
					false), _was_info_updated(false) {
#ifdef LOGS
		start_log_file(const_cast<char*>(_test_name.c_str()),
				const_cast<char*>(_test_info.c_str()));

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
		if (this->_was_error_updated == false)
			this->update_errors();
		if (this->_was_info_updated == false)
			this->update_infos();

#ifdef LOGS
		::end_log_file();

#ifdef BUILDPROFILER
		this->profiler_thread->end_profile();
#endif

#endif
	}

	void log_error_detail(std::string error_detail) {
		this->_error++;
		this->_was_error_updated = false;
#ifdef LOGS
		::log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void log_info_detail(std::string info_detail) {
		this->_info++;
		this->_was_info_updated = false;
#ifdef LOGS
		::log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	void start_iteration() {
		if (this->_was_error_updated == false)
			this->update_errors();
		if (this->_was_info_updated == false)
			this->update_infos();

		this->_error = 0;
		this->_info = 0;
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
		this->_was_error_updated = true;
		if (this->_error != 0) {
#ifdef LOGS
			::log_error_count(this->_error);
#endif
		}
	}

	void update_infos() {
		this->_was_info_updated = true;
		if (this->_info != 0) {
#ifdef LOGS
			::log_info_count(this->_info);
#endif
		}
	}

	void set_iter_interval_print(size_t print_interval) {
#ifdef LOGS
		::set_iter_interval_print(print_interval);
#endif
	}

	void set_max_errors_iter(size_t max_errors) {
#ifdef LOGS
		::set_max_errors_iter(max_errors);
#endif
	}

	void set_max_infos_iter(size_t max_errors) {
#ifdef LOGS
		::set_max_infos_iter(max_errors);
#endif
	}

	friend std::ostream& operator<<(std::ostream& os, Log& d) {
		std::string file_name = "No log file name, build with the libraries";
#ifdef LOGS
		file_name = get_log_file_name();
#endif
		os << "LOGFILENAME: " << file_name << std::endl;
		os << "Error: " << d._error << std::endl;
		os << "Info: " << d._info << std::endl;
		os << "Test info: " << d._test_info << std::endl;
		os << "Test name: " << d._test_name;
		return os;
	}

	uint64_t get_errors() {
		return this->_error;
	}

	uint64_t get_infos() {
		return this->_error;
	}
private:
	//Hide copy constructor to avoid copies
	//pass only as reference to the function/method
	Log(const Log& l) :
			_error(l._error), _info(l._info), _was_error_updated(
					l._was_error_updated), _was_info_updated(
					l._was_info_updated) {
	}

	uint64_t _error;
	uint64_t _info;
	std::string _test_name;
	std::string _test_info;

	bool _was_error_updated;
	bool _was_info_updated;

#ifdef LOGS
#ifdef BUILDPROFILER
	std::shared_ptr<Profiler> profiler_thread;
#endif
#endif

};

}

#endif /* LOG_H_ */
