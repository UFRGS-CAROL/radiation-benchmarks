/*
 * logs_processing.cpp
 *
 *  Created on: 12/10/2017
 *      Author: fernando
 */
#include "logs_processing.h"

#ifdef LOGS
#include "log_helper.h"
#endif

void compare_and_log() {
	int error_count = 0;
//			if (diff > MAX_ERROR_THRESHOLD) {
//				std::string error_detail = "position[" + std::to_string(i)
//						+ "][" + std::to_string(j) + "] expected: "
//						+ std::to_string(g) + " read: " + std::to_string(f);
//				error_count++;
//#ifdef LOGS
//				log_error_detail((char*)error_detail.c_str());
//
//#else
//				std::cout << error_detail << "\n";
//#endif
//			}

#ifdef LOGS
	log_error_count(error_count);
#endif
}


void start_benchmark(const char *gold_path, int n_reps, int n_gpu_threads,
		int n_gpu_blocks, int n_warmup, float alpha, const char* input_file_name,
		int in_size_i, int in_size_j){
#ifdef LOGS
#endif
}

void end_benchmark() {
#ifdef LOGS
	end_log_file();
#endif
}

void start_iteration_call() {
#ifdef LOGS
	start_iteration();
#endif
}

void end_iteration_call() {
#ifdef LOGS
	end_iteration();
#endif
}

void save_gold() {
}

void load_gold() {

}

