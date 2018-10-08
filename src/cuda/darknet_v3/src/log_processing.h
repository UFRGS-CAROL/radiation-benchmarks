/*
 * log_processing.h
 *
 *  Created on: 02/10/2018
 *      Author: fernando
 */

#ifndef LOG_PROCESSING_H_
#define LOG_PROCESSING_H_

#include <string>

#ifdef LOGS
#include "log_helper.h"
static const char* ABFT_TYPES[] = {"none", "abft"};
#endif

struct Log {
	static void start_log(std::string gold, int save_layer, int abft,
			int iterations, std::string app,
			unsigned char use_tensor_core_mode, int smx_red) {
#ifdef LOGS
		std::string test_info = std::string("gold_file: ") + gold;

		test_info += " save_layer: " + std::to_string(save_layer) + " abft_type: ";

		test_info += std::string(ABFT_TYPES[abft]) + " iterations: "
		+ std::to_string(iterations);

		test_info += " tensor_core_mode: "
		+ std::to_string(int(use_tensor_core_mode));

		test_info += " stream_redundancy: "
		+ std::to_string(smx_red);


		set_iter_interval_print(10);

		start_log_file(const_cast<char*>(app.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

	static void end_log() {
#ifdef LOGS
		end_log_file();
#endif
	}

	static void end_iteration_app() {
#ifdef LOGS
		end_iteration();
#endif
	}

	static void start_iteration_app() {
#ifdef LOGS
		start_iteration();
#endif
	}

	static void update_timestamp_app() {
#ifdef LOGS
		update_timestamp();
#endif
	}

	static void log_error_info(std::string error_detail) {
#ifdef LOGS
		log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	static void update_error_count(long error_count) {
#ifdef LOGS
		if(error_count)
		log_error_count(error_count);
#endif
	}
};

#endif /* LOG_PROCESSING_H_ */
