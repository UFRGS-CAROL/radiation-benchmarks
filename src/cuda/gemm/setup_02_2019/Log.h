/*
 * Log.h
 *
 *  Created on: Oct 4, 2018
 *      Author: carol
 */

#ifndef LOG_H_
#define LOG_H_
#include <string>
#include <sys/time.h>

#ifdef LOGS
#include "log_helper.h"
#endif

#include <string>

class Log {
public:
	bool generate;
	size_t size_matrices;
	int iterations;
	std::string a_input_path;
	std::string b_input_path;
	std::string c_input_path;
	std::string gold_inout_path;
	std::string precision;
	bool verbose;
	bool use_tensor_cores;
	bool triplicated;

	Log(int argc, char** argv, int input_size) {

		this->generate = this->find_int_arg(argc, argv, "--generate", 0);

		this->size_matrices = this->find_int_arg(argc, argv, "--size",
				input_size);

		this->iterations = this->find_int_arg(argc, argv, "--iterations", 1);

		this->a_input_path = this->find_char_arg(argc, argv, "--input_a",
				"./input_a.matrix");
		this->b_input_path = this->find_char_arg(argc, argv, "--input_b",
				"./input_b.matrix");
		this->c_input_path = this->find_char_arg(argc, argv, "--input_c",
				"./input_c.matrix");
		this->gold_inout_path = this->find_char_arg(argc, argv, "--gold",
				"./gold.matrix");

		this->precision = this->find_char_arg(argc, argv, "--precision",
				"float");

		this->use_tensor_cores = this->find_int_arg(argc, argv, "--tensor_cores", 0);

		this->verbose = this->find_int_arg(argc, argv, "--verbose", 0);

		this->triplicated = this->find_int_arg(argc, argv, "--triplicated", 0);

#ifdef LOGS
		std::string test_info = std::string(" iterations: ")
		+ std::to_string(this->iterations);

		test_info += " precision: " + this->precision;

		test_info += " matrix_n_dim: " + std::to_string(this->size_matrices);

		test_info += " triplicated: " + std::to_string(this->triplicated);

		std::string app = "gemm_tensor_cores_" + this->precision;
		set_iter_interval_print(10);

		start_log_file(const_cast<char*>(app.c_str()),
				const_cast<char*>(test_info.c_str()));
#endif
	}

	virtual ~Log() {
#ifdef LOGS
		end_log_file();
#endif
	}

	void end_iteration_app() {
#ifdef LOGS
		end_iteration();
#endif
	}

	void start_iteration_app() {
#ifdef LOGS
		start_iteration();
#endif
	}

	void update_timestamp_app() {
#ifdef LOGS
		update_timestamp();
#endif
	}

	void log_error(std::string error_detail) {
#ifdef LOGS
		log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
	}

	void log_info(std::string info_detail) {
#ifdef LOGS
		log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
	}

	void update_error_count(long error_count) {
#ifdef LOGS
		if (error_count)
		log_error_count(error_count);
#endif
	}

	 void update_info_count(long info_count) {
#ifdef LOGS
		if (info_count)
		log_info_count (info_count);
#endif
	}

	static double mysecond() {
		struct timeval tp;
		struct timezone tzp;
		int i = gettimeofday(&tp, &tzp);
		return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
	}

	static void del_arg(int argc, char **argv, int index) {
		int i;
		for (i = index; i < argc - 1; ++i)
			argv[i] = argv[i + 1];
		argv[i] = 0;
	}

	static int find_int_arg(int argc, char **argv, std::string arg, int def) {
		int i;
		for (i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				def = atoi(argv[i + 1]);
				del_arg(argc, argv, i);
				del_arg(argc, argv, i);
				break;
			}
		}
		return def;
	}

	static std::string find_char_arg(int argc, char **argv, std::string arg,
			std::string def) {
		int i;
		for (i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				def = std::string(argv[i + 1]);
				del_arg(argc, argv, i);
				del_arg(argc, argv, i);
				break;
			}
		}
		return def;
	}

	static int find_arg(int argc, char* argv[], std::string arg) {
		int i;
		for (i = 0; i < argc; ++i) {
			if (!argv[i])
				continue;
			if (std::string(argv[i]) == arg) {
				del_arg(argc, argv, i);
				return 1;
			}
		}
		return 0;
	}

	static float find_float_arg(int argc, char **argv, char *arg, float def) {
		int i;
		for (i = 0; i < argc - 1; ++i) {
			if (!argv[i])
				continue;
			if (0 == strcmp(argv[i], arg)) {
				def = atof(argv[i + 1]);
				del_arg(argc, argv, i);
				del_arg(argc, argv, i);
				break;
			}
		}
		return def;
	}

};

#endif /* LOG_H_ */
