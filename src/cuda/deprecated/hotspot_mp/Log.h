/*
 * Log.h
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#ifndef LOG_H_
#define LOG_H_
#include <string>
#include <sys/time.h>

#ifdef LOGS
#include "log_helper.h"
#endif

#include <string>

struct Log {
	bool to_log;
	double tic;
	double toc;
	size_t error_count;
	size_t info_count;

	Log(std::string& app, std::string& test_info, bool to_log);
	Log();
	Log(const Log& a);

	virtual ~Log();

	static void force_end(std::string& error);

	void end_iteration_app();

	void start_iteration_app();

	void update_timestamp_app();

	void log_error(std::string error_detail);

	void log_info(std::string info_detail);

	void update_error_count();

	void update_info_count();

	double iteration_time();

	static double mysecond();
	void fatal(std::string& s);

	Log& operator=(const Log& other);

};

#endif /* LOG_H_ */
