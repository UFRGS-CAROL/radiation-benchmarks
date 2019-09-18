/*
 * Log.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef LOG_H_
#define LOG_H_

#include "Parameters.h"
#include "log_helper.h"

#include <stdexcept>
#include <string>

static void errorat(const char* error_string, const char* file, unsigned line) {

	std::string err_(error_string);
	err_ += std::string("\nERROR AT: ") + std::string(file) + ":"
			+ std::to_string(line);
	throw std::runtime_error(err_);
}

#define fatalerror(str) errorat(str, __FILE__, __LINE__)

struct Log {
	uint64 error;
	uint64 info;
	std::string test_name;
	std::string test_info;

	Log(const Log& l) :
			error(l.error), info(l.info) {
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

	void update_errors(uint64 errors) {
		if (errors != 0) {
#ifdef LOGS
			::log_error_count(errors);
#endif
		}
	}

	void update_infos(uint64 infos) {
		if (infos != 0) {
#ifdef LOGS
			::log_info_count(infos);
#endif
		}
	}
};

#endif /* LOG_H_ */
