/*
 * log_processing.h
 *
 *  Created on: 30/04/2017
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include <sys/time.h> //cont time
#include "network.h" //save layer
#include "layer.h" //save layer
#include "box.h" //boxes
#include <fstream>
#include <vector>
#include <string>
#include "helpful.h"
#include <cmath>
#include <iostream>

#ifdef LOGS
#include "log_helper.h"
#endif


class Log {
public:
	/**
	 * get_image_filenames are used by generate
	 */
	const std::vector<std::string> abft_types = {"none", "abft"};


	Log(std::string gold, int save_layer, int abft, int iterations,
			std::string app, unsigned char use_tensor_core_mode) {
#ifdef LOGS
		std::string test_info = std::string("gold_file: ") + gold;

		test_info += " save_layer: " + std::to_string(save_layer) + " abft_type: ";


		test_info += this.abft_types[abft] + " iterations: " + std::to_string(iterations);

		test_info += " tensor_core_mode: " + std::to_string(int(use_tensor_core_mode));

		set_iter_interval_print(10);

		start_log_file(const_cast<char*>(app.c_str()), const_cast<char*>(test_info.c_str()));
#endif
	}

	virtual ~Log() {
#ifdef LOGS
		end_log_file();
#endif
	}

	void end_iteration_app() {
#ifdef LOGS
		end_iteration();
#endif
	}

	void start_iteration_app() {
#ifdef LOGS
		start_iteration();
#endif
	}

	void update_timestamp_app() {
#ifdef LOGS
		update_timestamp();
#endif
	}

	std::string get_small_log_file() {
		std::string temp(get_log_file_name());
		std::vector < std::string > ret_array = split(temp, '/');
		std::string str_ret = ret_array[ret_array.size() - 1];
		return str_ret;
	}

};

#endif /* LOG_PROCESSING_H_ */
