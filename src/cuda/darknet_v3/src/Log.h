/*
 * log_processing.h
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include <string>
#include <iostream>


#ifdef LOGS
#include "log_helper.h"
#endif

class Log {
public:
	/**
	 * get_image_filenames are used by generate
	 */

	Log(std::string gold, int save_layer, int abft, int iterations,
			std::string app, unsigned char use_tensor_core_mode);

	virtual ~Log();

	void end_iteration_app();

	void start_iteration_app();

	void update_timestamp_app();

	std::string get_small_log_file();

};

#endif /* LOG_PROCESSING_H_ */
