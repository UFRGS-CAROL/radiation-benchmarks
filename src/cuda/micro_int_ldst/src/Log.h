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

struct Log {
	uint64_t error;
	uint64_t info;
	std::string test_name;
	std::string test_info;

	Log(const Log& l);

	Log();
	Log(std::string& test_name, std::string& test_info);

	friend std::ostream& operator<<(std::ostream& os, Log& d);

	virtual ~Log();

	void log_error_detail(std::string error_detail);

	void log_info_detail(std::string info_detail);

	void start_iteration();
	void end_iteration();

	void update_errors();

	void update_infos();

	void update_errors(uint64_t errors);

	void update_infos(uint64_t infos);
};

#endif /* LOG_H_ */
