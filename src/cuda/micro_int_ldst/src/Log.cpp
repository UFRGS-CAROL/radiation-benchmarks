/*
 * Log.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include <iostream>

#include "Log.h"

Log::Log(const Log& l) :
			error(l.error), info(l.info) {
	}

Log::Log() :
			error(0), info(0) {
	}

Log::Log(std::string& test_name, std::string& test_info) :
			error(0), info(0), test_name(test_name), test_info(test_info) {
#ifdef LOGS
		start_log_file(const_cast<char*>(test_name.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

std::ostream& operator<<(std::ostream& os, Log& d) {
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

	Log::~Log() {
#ifdef LOGS
		::end_log_file();
#endif
	}

	void Log::log_error_detail(std::string error_detail) {
		this->error++;
#ifdef LOGS
		::log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void Log::log_info_detail(std::string info_detail) {
		this->info++;
#ifdef LOGS
		::log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	void Log::start_iteration() {
		this->error = 0;
		this->info = 0;
#ifdef LOGS
		::start_iteration();
#endif
	}

	void Log::end_iteration() {
#ifdef LOGS
		::end_iteration();
#endif
	}

	void Log::update_errors() {
		if (this->error != 0) {
#ifdef LOGS
			::log_error_count(this->error);
#endif
		}
	}

	void Log::update_infos() {
		if (this->info != 0) {
#ifdef LOGS
			::log_info_count(this->info);
#endif
		}
	}

	void update_errors(uint64_t errors) {
		if (errors != 0) {
#ifdef LOGS
			::log_error_count(errors);
#endif
		}
	}

	void update_infos(uint64_t infos) {
		if (infos != 0) {
#ifdef LOGS
			::log_info_count(infos);
#endif
		}
	}

