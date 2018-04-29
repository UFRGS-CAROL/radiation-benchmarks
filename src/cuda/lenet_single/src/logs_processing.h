/*
 * LogsProcessing.h
 *
 *  Created on: 16/07/2017
 *      Author: fernando
 */

#ifndef LOGSPROCESSING_H_
#define LOGSPROCESSING_H_

#define RADIATION_PARAMETERS 3

class LogsProcessing{
private:
	float layer_threshold_error = 1e-5;
	char *save_layer_data = "/var/radiation-benchmarks/data";
	bool generate;

public:
	LogsProcessing();
	void start_count_app(char *test, char *app);

	void finish_count_app();

	void start_iteration_app();

	void end_iteration_app();

	void inc_count_app();

	void log_error_app(char *error_detail);

	/**
	 * support function only to check if two layers have
	 * the same value
	 */
	bool compare_layer(float *l1, float *l2, int n);

	bool compare_output();

	void* load_gold_layers(int img, int layer_size);

	void save_gold_layers();

	void compare_and_save_layers();
	char* get_log_filename();
};
#endif /* LOGSPROCESSING_H_ */
