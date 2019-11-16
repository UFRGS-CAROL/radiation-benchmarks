/*
 * Log.h
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#ifndef LOG_H_
#define LOG_H_
#include <sys/time.h>
#include <string>
#include <iostream>

#ifdef LOGS
#include "log_helper.h"
#endif

struct Log {

	Log(int argc, char** argv);

	friend std::ostream& operator<<(std::ostream& os, const Log& log_obj);

	virtual ~Log();

	void end_iteration();

	void start_iteration();

	void update_timestamp();
	void log_error(std::string error_detail);
	void log_info(std::string info_detail);
	void update_error_count(long error_count);
	void update_info_count(long info_count);

	bool generate;
	size_t size_matrices;
	int iterations;
	std::string a_input_path;
	std::string b_input_path;
	std::string c_input_path;
	std::string gold_inout_path;

	std::string precision;
	bool verbose;
	bool use_tensor_cores;
	bool triplicated;
	std::string dmr;
	double alpha;
	double beta;
	uint32_t check_block;

private:

	void del_arg(int argc, char **argv, int index);

	int find_int_arg(int argc, char **argv, std::string arg, int def);

	std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def);

	int find_arg(int argc, char* argv[], std::string arg);
	float find_float_arg(int argc, char **argv, std::string arg, float def);
};

#endif /* LOG_H_ */
