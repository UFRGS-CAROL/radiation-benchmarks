#include "Log.h"

#include "common.h"

Log::Log(int argc, char** argv) :
		alpha(1), beta(0) {

	//getting alpha and beta
	this->alpha = this->find_float_arg(argc, argv, "--alpha", 1);
	this->beta = this->find_float_arg(argc, argv, "--beta", 0);

	this->generate = this->find_int_arg(argc, argv, "--generate", 0);

	this->size_matrices = this->find_int_arg(argc, argv, "--size", 1024);

	this->iterations = this->find_int_arg(argc, argv, "--iterations", 1);

	this->a_input_path = this->find_char_arg(argc, argv, "--input_a",
			"./input_a.matrix");
	this->b_input_path = this->find_char_arg(argc, argv, "--input_b",
			"./input_b.matrix");
	this->c_input_path = this->find_char_arg(argc, argv, "--input_c",
			"./input_c.matrix");
	this->gold_inout_path = this->find_char_arg(argc, argv, "--gold",
			"./gold.matrix");


	this->precision = this->find_char_arg(argc, argv, "--precision", "float");

	this->dmr = this->find_char_arg(argc, argv, "--dmr", "none");

	this->use_tensor_cores = this->find_int_arg(argc, argv, "--tensor_cores",
			0);

	this->check_block = this->find_int_arg(argc, argv, "--opnum", BLOCK_SIZE);

	this->verbose = this->find_int_arg(argc, argv, "--verbose", 0);

	this->triplicated = this->find_int_arg(argc, argv, "--triplicated", 0);

#ifdef LOGS
	std::string test_info = std::string(" iterations: ")
	+ std::to_string(this->iterations);

	test_info += " precision: " + this->precision;

	test_info += " matrix_n_dim: " + std::to_string(this->size_matrices);

	test_info += " triplicated: " + std::to_string(this->triplicated);

	test_info += " dmr: " + this->dmr;

	test_info += " use_tensorcores: " + std::to_string(this->use_tensor_cores);

	test_info += " checkblock: " + std::to_string(this->check_block);

	test_info += " alpha: " + std::to_string(this->alpha);
	test_info += " beta: " + std::to_string(this->beta);

	std::string app = "gemm_tensor_cores_" + this->precision;
	set_iter_interval_print(10);

	start_log_file(const_cast<char*>(app.c_str()),
			const_cast<char*>(test_info.c_str()));
#endif
}

std::ostream& operator<<(std::ostream& os, const Log& log_obj) {
	os << "Generate: " << log_obj.generate << std::endl;
	os << "A input path: " << log_obj.a_input_path << std::endl;
	os << "B input path: " << log_obj.b_input_path << std::endl;
	os << "C input path: " << log_obj.c_input_path << std::endl;
	os << "Gold in/out path: " << log_obj.gold_inout_path << std::endl;
	os << "Iterations: " << log_obj.iterations << std::endl;
	os << "Matrix size: " << log_obj.size_matrices << std::endl;
	os << "Precision: " << log_obj.precision << std::endl;
	os << "Verbose: " << log_obj.verbose << std::endl;
	os << "DMR type: " << log_obj.dmr << std::endl;
	os << "Tensor cores: " << log_obj.use_tensor_cores << std::endl;
	os << "Alpha: " << log_obj.alpha << std::endl;
	os << "Beta: " << log_obj.beta << std::endl;
	os << "DMR Block checking " << log_obj.check_block << std::endl;
	os << "LOGFILENAME: " << ::get_log_file_name();
	return os;
}

Log::~Log() {
#ifdef LOGS
	end_log_file();
#endif
}

void Log::end_iteration() {
#ifdef LOGS
	::end_iteration();
#endif
}

void Log::start_iteration() {
#ifdef LOGS
	::start_iteration();
#endif
}

void Log::update_timestamp() {
#ifdef LOGS
	::update_timestamp();
#endif
}

void Log::log_error(std::string error_detail) {
#ifdef LOGS
	log_error_detail(const_cast<char*>(error_detail.c_str()));
#endif
}

void Log::log_info(std::string info_detail) {
#ifdef LOGS
	log_info_detail(const_cast<char*>(info_detail.c_str()));
#endif
}

void Log::update_error_count(long error_count) {
#ifdef LOGS
	if (error_count)
	log_error_count(error_count);
#endif
}

void Log::update_info_count(long info_count) {
#ifdef LOGS
	if (info_count)
	log_info_count (info_count);
#endif
}

void Log::del_arg(int argc, char **argv, int index) {
	int i;
	for (i = index; i < argc - 1; ++i)
		argv[i] = argv[i + 1];
	argv[i] = 0;
}

int Log::find_int_arg(int argc, char **argv, std::string arg, int def) {
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

std::string Log::find_char_arg(int argc, char **argv, std::string arg,
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

int Log::find_arg(int argc, char* argv[], std::string arg) {
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

float Log::find_float_arg(int argc, char **argv, std::string arg, float def) {
	int i;
	for (i = 0; i < argc - 1; ++i) {
		if (!argv[i])
			continue;
		if (std::string(argv[i]) == arg) {
			def = atof(argv[i + 1]);
			del_arg(argc, argv, i);
			del_arg(argc, argv, i);
			break;
		}
	}
	return def;
}
