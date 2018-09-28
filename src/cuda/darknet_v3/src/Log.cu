/*
 * Log.cu
 *
 *  Created on: 27/09/2018
 *      Author: fernando
 */

#include "Log.h"
#include <cmath>
#include <fstream>
#include <vector>
#include "helpful.h"



static const char* ABFT_TYPES[] = { "none", "abft" };


Log::Log(std::string gold, int save_layer, int abft, int iterations,
		std::string app, unsigned char use_tensor_core_mode) {
#ifdef LOGS
	std::string test_info = std::string("gold_file: ") + gold;

	test_info += " save_layer: " + std::to_string(save_layer) + " abft_type: ";

	test_info += std::string(ABFT_TYPES[abft]) + " iterations: " + std::to_string(iterations);

	test_info += " tensor_core_mode: " + std::to_string(int(use_tensor_core_mode));

	set_iter_interval_print(10);

	start_log_file(const_cast<char*>(app.c_str()), const_cast<char*>(test_info.c_str()));
#endif
}

Log::~Log() {
#ifdef LOGS
	end_log_file();
#endif
}

void Log::end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}

void Log::start_iteration_app() {
#ifdef LOGS
	start_iteration();
#endif
}

void Log::update_timestamp_app() {
#ifdef LOGS
	update_timestamp();
#endif
}

std::string Log::get_small_log_file() {
	std::string temp(get_log_file_name());
	std::vector < std::string > ret_array = split(temp, '/');
	std::string str_ret = ret_array[ret_array.size() - 1];
	return str_ret;
}

void Log::log_error_info(char* error_detail){
#ifdef LOGS
			log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
}

void Log::update_error_count(long error_count){
#ifdef LOGS
	log_error_count(error_count);
#endif
}
