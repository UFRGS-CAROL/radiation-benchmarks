/*
 * logs_processing.h
 *
 *  Created on: 12/10/2017
 *      Author: fernando
 */

#ifndef LOGS_PROCESSING_H_
#define LOGS_PROCESSING_H_
#include <string>
#include "common.h"

#define MAX_ERROR_THRESHOLD 1e-8

int compare_and_log(XYZ *found, XYZ *gold, int RESOLUTIONI, int RESOLUTIONJ);

void start_benchmark(std::string gold_path, int n_reps, int n_gpu_threads,
		int n_gpu_blocks, int n_warmup, float alpha,
		std::string input_file_name, int in_size_i, int in_size_j,
		int out_size_i, int out_size_j);
void end_benchmark();
void start_iteration_call();
void end_iteration_call();

void save_gold(XYZ *gold, int size, std::string gold_path);

void load_gold(XYZ* gold, int size, std::string gold_path);

#endif /* LOGS_PROCESSING_H_ */
