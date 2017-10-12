/*
 * logs_processing.h
 *
 *  Created on: 12/10/2017
 *      Author: fernando
 */

#ifndef LOGS_PROCESSING_H_
#define LOGS_PROCESSING_H_

void compare_and_log();
//	start_benchmark(p.gold_in_out.c_str(), p.n_reps, p.n_gpu_threads, p.n_gpu_blocks, p.n_warmup, p.alpha, p.file_name, p.in_size_i, p.in_size_j);

void start_benchmark(const char *gold_path, int n_reps, int n_gpu_threads,
		int n_gpu_blocks, int n_warmup, float alpha, const char* input_file_name,
		int in_size_i, int in_size_j);
void end_benchmark();
void start_iteration_call();
void end_iteration_call();
void save_gold();
void load_gold();

#endif /* LOGS_PROCESSING_H_ */
