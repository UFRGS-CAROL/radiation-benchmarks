/*
 * Log.h
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_
#include <sys/time.h>
#include <string>
#include <iostream>
#include <memory>

#include "include/generic_log.h"

struct Parameters {

	Parameters(int argc, char** argv);

	friend std::ostream& operator<<(std::ostream& os,
			const Parameters& log_obj);

	virtual ~Parameters() = default;

	void end_iteration();

	void start_iteration();

	void usage(char **argv);

//	void update_timestamp();
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
	float alpha;
	float beta;
	bool use_cublas;
	bool use_cutlass;

	uint32_t check_block;

	bool check_input_existence;

private:
	std::shared_ptr<rad::Log> log;
};

#endif /* PARAMETERS_H_ */
