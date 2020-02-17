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
#endif

#include <string>
#include <iostream>

namespace rad {

struct Log {
	uint64_t error;
	uint64_t info;
	std::string test_name;
	std::string test_info;

	bool was_info_updated;
	bool was_error_updated;

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

	Log(const Log& l) :
			error(l.error), info(l.info), was_error_updated(false), was_info_updated(
					false) {
	}

	Log() :
			error(0), info(0), was_error_updated(false), was_info_updated(false) {
	}

	Log(std::string& test_name, std::string& test_info) :
			error(0), info(0), test_name(test_name), test_info(test_info), was_error_updated(
					false), was_info_updated(false) {
#ifdef LOGS
		start_log_file(const_cast<char*>(test_name.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

	~Log() {
		if (this->was_error_updated == false)
			this->update_errors();
		if (this->was_info_updated == false)
			this->update_infos();

#ifdef LOGS
		::end_log_file();
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

};

}

#endif /* LOG_H_ */
