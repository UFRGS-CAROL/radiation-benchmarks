/*
 * logs_processing.h
 *
 *  Created on: 12/10/2017
 *      Author: fernando
 */

#ifndef LOGS_PROCESSING_H_
#define LOGS_PROCESSING_H_

#include <string>
#include <vector>
#include <atomic>

#define MAX_ERROR_THRESHOLD 1e-8

int compare_and_log(std::pair<int*, std::atomic_int*> gold,
		std::atomic_int* found, int size, long it_cpu, long it_gpu);

void start_benchmark(std::string gold_file, std::string input_file,
		int n_gpu_threads, int n_gpu_blocks, int n_threads, int n_warmup,
		int n_reps, int switching_limit);
void end_benchmark();
void start_iteration_call();
void end_iteration_call();

void save_gold(std::atomic_int* gold, int size, std::string gold_path);

void load_gold(std::pair<int*, std::atomic_int*> gold, int size, std::string gold_path);

double get_time();

#endif /* LOGS_PROCESSING_H_ */
