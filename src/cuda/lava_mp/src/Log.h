/*
 * Log.h
 *
 *  Created on: 29/09/2019
 *      Author: fernando
 */

#ifndef LOG_H_
#define LOG_H_

#ifdef LOGS
#include "log_helper.h"
#endif

#include <stdexcept>
#include <string>

struct Log {
	size_t error;
	size_t info;
	std::string test_name;
	std::string test_info;

	Log(const Log& l) :
			error(l.error), info(l.info), test_info(l.test_info), test_name(
					l.test_name) {
	}

	Log() :
			error(0), info(0) {
	}

	Log(std::string& test_name, std::string& test_info) :
			error(0), info(0), test_name(test_name), test_info(test_info) {
#ifdef LOGS
		start_log_file(const_cast<char*>(test_name.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

	Log& operator=(const Log& rhs) {
		this->error = rhs.error;
		this->info = rhs.info;
		this->test_info = rhs.test_info;
		this->test_name = rhs.test_name;
		return *this;
	}

	friend std::ostream& operator<<(std::ostream& os, Log& d) {
		std::string file_name = "No log file name, build with the libraries";
#ifdef LOGS
		file_name = get_log_file_name();
#endif
		os << "Logfilename: " << file_name << std::endl;
		os << "Error: " << d.error << std::endl;
		os << "Info: " << d.info << std::endl;
		os << "Test info: " << d.test_info << std::endl;
		os << "Test name: " << d.test_name;
		return os;
	}

	virtual ~Log() {
#ifdef LOGS
		::end_log_file();
#endif
	}

	std::string get_log_file_name() {
#ifdef LOGS
		return std::string(::get_log_file_name());
#else
		return "";
#endif
	}

	void set_max_errors_iter(size_t max_errors) {
#ifdef LOGS
		::set_max_errors_iter(max_max_errors);
#endif
	}

	void log_error_detail(std::string error_detail) {
		this->error++;
#ifdef LOGS
		::log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void log_info_detail(std::string info_detail) {
		this->info++;
#ifdef LOGS
		::log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	void start_iteration() {
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
		if (this->error != 0) {
#ifdef LOGS
			::log_error_count(this->error);
#endif
		}
	}

	void update_infos() {
		if (this->info != 0) {
#ifdef LOGS
			::log_info_count(this->info);
#endif
		}
	}

	void update_errors(size_t errors) {
		if (errors != 0) {
#ifdef LOGS
			::log_error_count(errors);
#endif
		}
	}

	void update_infos(size_t infos) {
		if (infos != 0) {
#ifdef LOGS
			::log_info_count(infos);
#endif
		}
	}
};

#endif /* LOG_H_ */
