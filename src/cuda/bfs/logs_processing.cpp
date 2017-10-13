/*
 * logs_processing.cpp
 *
 *  Created on: 12/10/2017
 *      Author: fernando
 */

#include "logs_processing.h"
#include <cmath>
#include <fstream>
#include <assert.h>
#include <iostream>
#include <sys/time.h>

#ifdef LOGS
#include "log_helper.h"
#endif

// Returns the current system time in microseconds
double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double(tv.tv_sec * 1000000) + double(tv.tv_usec)) / 1000000;
}

int compare_and_log(std::pair<int*, std::atomic_int*> gold,
		std::atomic_int* found, int size, long it_cpu, long it_gpu) {
	int errors = 0;
	// cost of nodes in the output

	for (int i = 0; i < size; i++) {
		int g_index = gold.first[i];
		int f_index = i;

		int g_cost = gold.second[i];
		int f_cost = found[i].load();
		if (g_index != f_index || g_cost != f_cost) {
			errors++;
			//printf("Computed node %ld cost (%ld != %ld) does not match the expected value\n", i, h_cost[i].load(), cost);
#ifdef LOGS
			char error_detail[250];
			sprintf(error_detail,"Node: %d index_e: %d index_r: %d cost_e: %d cost_r: %d CPU: %ld GPU: %ld\n",
					i, g_index, i, g_cost, f_cost, it_cpu, it_gpu);
			log_error_detail(error_detail);
#endif

		}
	}

#ifdef LOGS
	log_error_count(errors);
#endif
	return errors;
}

void start_benchmark(std::string gold_file, std::string input_file,
		int n_gpu_threads, int n_gpu_blocks, int n_threads, int n_warmup,
		int n_reps, int switching_limit) {
#ifdef LOGS
	std::string header = "gold: " + gold_file + " input_file: " + input_file + " n_gpu_threads: " +
	std::to_string(n_gpu_threads) + " n_gpu_blocks: " + std::to_string(n_gpu_blocks) +
	" n_threads: " + std::to_string(n_threads) + " n_warmup: " + std::to_string(n_warmup) +
	" n_reps: " + std::to_string(n_reps) + " switching_limit: " + std::to_string(switching_limit);

	start_log_file("cudaHeterogeneousBFS", (char*) header.c_str());
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

void save_gold(std::atomic_int* gold, int size, std::string gold_path) {
	std::vector<int> indexes(size);

	std::vector<int> to_write(size);
	for (int i = 0; i < size; i++) {
		to_write[i] = gold[i].load();
		indexes[i] = i;
	}

	std::ofstream gold_out(gold_path, std::ios::out | std::ios::binary);
	assert(gold_out.is_open() && "Gold file not opened for writing");
	gold_out.write(reinterpret_cast<char*>(&size), sizeof(int));
	gold_out.write(reinterpret_cast<char*>(indexes.data()), sizeof(int) * size);

	gold_out.write(reinterpret_cast<char*>(to_write.data()),
			sizeof(int) * size);
	gold_out.close();
}

void load_gold(std::pair<int*, std::atomic_int*> gold, int size,
		std::string gold_path) {
	std::ifstream gold_in(gold_path, std::ios::in | std::ios::binary);
	assert(gold_in.is_open() && "Gold file not opened for reading");
	int temp_size;

	gold_in.read(reinterpret_cast<char*>(&temp_size), sizeof(int));
	assert(size == temp_size && "Gold Size is not ok");

	//indexes
	gold_in.read(reinterpret_cast<char*>(gold.first), sizeof(int) * size);

	//cost vector
	std::vector<int> to_read(size);
	gold_in.read(reinterpret_cast<char*>(to_read.data()), sizeof(int) * size);
	for (int i = 0; i < size; i++) {
		gold.second[i] = to_read[i];
	}

	gold_in.close();
}

