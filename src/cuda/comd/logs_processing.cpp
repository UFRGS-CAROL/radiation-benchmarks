/*
 * logs_processing.cpp
 *
 *  Created on: 07/09/2017
 *      Author: fernando
 */

#include "logs_processing.h"

#ifdef LOGS
#include "log_helper.h"
#endif




void start_count_app(char *gold_file, int iterations) {
#ifdef LOGS
	std::string header = "gold_file: " + std::string(gold_file) + " iterations: " + std::to_string(iterations);
	start_log_file("comd", header.c_std());
#endif
}

void finish_count_app() {
#ifdef LOGS
	end_log_file();
#endif
}

void start_iteration_app() {
#ifdef LOGS
	start_iteration();
#endif
}

void end_iteration_app() {
#ifdef LOGS
	end_iteration();
#endif
}


