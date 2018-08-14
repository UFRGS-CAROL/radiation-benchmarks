/*
 * LogHelper.h
 *
 *  Created on: 13/08/2018
 *      Author: fernando
 */

#ifndef LOGHELPER_H_
#define LOGHELPER_H_

#include <string>


class LogHelper {

	std::string timestamp_watchdog;
	std::string timestamp_file;
	std::string vardir_key;

	// Max errors that can be found for a single iteration
	// If more than max errors is found, exit the program
	size_t max_errors_per_iter;
	size_t max_infos_per_iter;

	// Absolute path for log file, if needed
	std::string absolute_path;

	std::string logdir_key;
	std::string signalcmd_key;

	std::string config_file;

	// Used to print the log only for some iterations, equal 1 means print every iteration
	int iter_interval_print;

	// Used to log max_error_per_iter details each iteration
	int log_error_detail_count;
	int log_info_detail_count;

	std::string log_file_name;
	std::string full_log_file_name;

	// Saves the last amount of error found for a specific iteration
	size_t last_iter_errors;
	// Saves the last iteration index that had an error
	size_t last_iter_with_errors;

	size_t kernels_total_errors;
	size_t kernels_total_infos;
	size_t iteration_number;

	double kernel_time_acc;
	double kernel_time;

	long long it_time_start;

	/**
	 * Private Methods
	 */

	// Functions to check ECC
	/**
	 * String contains
	 * check if word contains in sent string
	 * return 1 if contains
	 * return 0 otherwise
	 */
	int contains(char *sent, const char *word);

	/**
	 * popen_call
	 * call popen and check if check_line is in output string
	 * if check_line is in popen output an output is writen in output_line
	 * return 1 if the procedure executed
	 * return 0 otherwise
	 */
	int popen_call(std::string cmd, std::string check_line);

	/**
	 * This functions checks if ECC is enable or disabled for NVIDIA GPUs
	 * 0 if ECC is disabled
	 * 1 if ECC is enabled
	 */
	int check_ecc_status();

	/**
	 * Get current time
	 */
	long long get_time();

	/**
	 * Read config file to get the value of a 'key = value' pair
	 */
	std::string get_value_config(std::string key);

	/**
	 * Start log file
	 */
	void start_log_file(std::string benchmark_name, std::string test_info);

	/**
	 * End log file
	 */
	void end_log_file();

public:
	/**
	 * Public methods
	 */

	/**
	 * Set max errors per iteration
	 */
	void set_max_errors_iter(size_t max_errors);

	/**
	 * Set max info per iteration
	 */
	void set_max_infos_iter(size_t max_infos);

	/**
	 * Set the interval the program must print log details, default is 1 (each iteration)
	 */
	void set_iter_interval_print(int interval);

	/**
	 * Update with current timestamp the file where the software watchdog watchs
	 */
	void update_timestamp();

	/**
	 *  Return the name of the log file generated
	 */
	std::string get_log_file_name();

	/**
	 * Start time to measure kernel time, also update iteration number and log to file
	 */
	void start_iteration();

	/**
	 * Finish the measured kernel time log both time (total time and kernel time)
	 */
	void end_iteration();

	/**
	 * Update total errors variable and log both errors(total errors and kernel errors)
	 */
	void log_error_count(size_t kernel_errors);

	/**
	 * Update total infos variable and log both infos(total infos and iteration infos)
	 */
	void log_info_count(size_t info_count);

	/**
	 * Print some string with the detail of an error to log file
	 */
	void log_error_detail(std::string string);

	/**
	 *  Print some string with the detail of an error/information to log file
	 */
	void log_info_detail(std::string string);

	/**
	 * Get the iteration number
	 */
	size_t get_iteration_number();

	/**
	 * Generate the log file name, log info from user about the test to be executed and reset log variables
	 * OLD start_log_file
	 */
	LogHelper(std::string benchmark_name, std::string test_info);

	/**
	 * Log the string "#END" and reset global variables
	 */
	virtual ~LogHelper();

};


#endif /* LOGHELPER_H_ */
