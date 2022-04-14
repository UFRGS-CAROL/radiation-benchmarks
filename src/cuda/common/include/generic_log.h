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
#include <utility>

namespace rad {

#define DEFAULT_MAX_ERRORS_ITERATION_COUNT 500
#define DEFAULT_MAX_INFOS_ITERATION_COUNT 500

struct Log {
	Log() :
			_error(0), _info(0), _was_error_updated(false), _was_info_updated(false), _max_errors_per_iteration(
					0), _max_infos_per_iteration(0) {
	}

	Log(std::string test_name, std::string test_info, size_t print_interval = 1,
			size_t max_errors_iteration = DEFAULT_MAX_ERRORS_ITERATION_COUNT,
			size_t max_infos_iteration = DEFAULT_MAX_INFOS_ITERATION_COUNT) :
			_error(0), _info(0), _test_name(std::move(test_name)), _test_info(std::move(test_info)), _was_error_updated(
					false), _was_info_updated(false), _max_errors_per_iteration(
					max_errors_iteration), _max_infos_per_iteration(max_infos_iteration) {
#ifdef LOGS
		start_log_file(const_cast<char*>(_test_name.c_str()),
				const_cast<char*>(_test_info.c_str()));

		::set_iter_interval_print(print_interval);
		//set the max errors per iterations
		::set_max_errors_iter(max_errors_iteration);
		//set the max infos per iterations
		::set_max_infos_iter(max_infos_iteration);

		/**
		 * Profiler macros to define profiler thread
		 * must build libraries before build with this macro
		 */
#ifdef BUILDPROFILER
		//      std::string log_file_name(get_log_file_name());
		this->profiler_thread = std::make_shared<NVMLWrapper>(0, this->get_log_file_name());

		//START PROFILER THREAD
		profiler_thread->start_profile();
#endif
#endif
	}

	~Log() {
		if (!this->_was_error_updated)
			this->update_errors();
		if (!this->_was_info_updated)
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
		if (this->_error < this->_max_errors_per_iteration) {
#ifdef LOGS
			::log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
		}
	}

	void log_info_detail(std::string info_detail) {
		this->_info++;
		this->_was_info_updated = false;
		if (this->_info < this->_max_infos_per_iteration) {
#ifdef LOGS
			::log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
		}
	}

	void start_iteration() {
		if (!this->_was_error_updated)
			this->update_errors();
		if (!this->_was_info_updated)
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

	friend std::ostream &operator<<(std::ostream &os, Log &d) {
		std::string file_name = "No log file name, build with the libraries";
#ifdef LOGS
		file_name = d.get_log_file_name();
#endif
		os << "LOGFILENAME: " << file_name << std::endl;
		os << "Error: " << d._error << std::endl;
		os << "Info: " << d._info << std::endl;
		os << "Test info: " << d._test_info << std::endl;
		os << "Test name: " << d._test_name << std::endl;
		os << "Max errors per iteration: " << d._max_errors_per_iteration << std::endl;
		os << "Max infos per iteration: " << d._max_infos_per_iteration;
		return os;
	}

	uint64_t get_errors() {
		return this->_error;
	}

	uint64_t get_infos() {
		return this->_error;
	}

	std::string get_log_file_name() {
	    char tmp[1024] = {'\0'};
#ifndef LOGS
		return "";
#else
        ::get_log_file_name(tmp);
        std::string return_string(tmp);
		return return_string;
#endif
	}

	/*
	 * There is no need for update_errors and update_infos to be public
	 */
	void update_errors() {
		this->_was_error_updated = true;
		if (this->_error != 0) {
			this->_error =
					(this->_error < this->_max_errors_per_iteration) ?
							this->_error : this->_max_errors_per_iteration - 1;
#ifdef LOGS
			::log_error_count(this->_error);
#endif
		}
	}

	void update_infos() {
		this->_was_info_updated = true;
		if (this->_info != 0) {
			this->_info =
					(this->_info < this->_max_infos_per_iteration) ?
							this->_info : this->_max_infos_per_iteration - 1;

#ifdef LOGS
			::log_info_count(this->_info);
#endif
		}
	}

private:
	//Hide copy constructor to avoid copies
	//pass only as reference to the function/method
	Log(const Log &l) :
			_error(l._error), _info(l._info), _was_error_updated(l._was_error_updated), _was_info_updated(
					l._was_info_updated), _max_errors_per_iteration(l._max_errors_per_iteration), _max_infos_per_iteration(
					l._max_infos_per_iteration) {
	}

	uint64_t _error = 0;
	uint64_t _info = 0;
	std::string _test_name;
	std::string _test_info;

	bool _was_error_updated = false;
	bool _was_info_updated = false;

	//Max errors and info lines per iteration
	const uint64_t _max_errors_per_iteration;
	const uint64_t _max_infos_per_iteration;

#ifdef LOGS
#ifdef BUILDPROFILER
	std::shared_ptr<Profiler> profiler_thread;
#endif
#endif

	/**
	 * Only the constructor is necessary from now
	 */
//  void set_iter_interval_print(size_t print_interval) {
//#warning "set_iter_interval_print is not allowed anymore, use the constructor to set the interval"
//  }
//
//  void set_max_errors_iter(size_t max_errors) {
//#warning "set_max_errors_iter is not allowed anymore, use the constructor to set the interval"
//  }
//
//  void set_max_infos_iter(size_t max_errors) {
//#warning "set_max_infos_iter is not allowed anymore, use the constructor to set the interval"
//  }
};

}

#endif /* LOG_H_ */
