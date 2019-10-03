//============================================================================
//	UPDATE
//============================================================================

//	14 APR 2011 Lukasz G. Szafaryn
//  2014-2018 Caio Lunardi
//  2018 Fernando Fernandes dos Santos

#include <iostream>

#ifdef LOGS

#include "log_helper.h"

#ifdef BUILDPROFILER

#ifdef FORJETSON
#include "include/JTX2Inst.h"
#define OBJTYPE JTX2Inst
#else
#include "include/NVMLWrapper.h"
#define OBJTYPE NVMLWrapper
#endif // FORJETSON

#endif // BUILDPROFILER

#endif // LOGHELPER

#include "setup_template.h"

void choose_setup_option(Parameters& parameters, Log& log) {
	if (parameters.precision == HALF) {
		if (parameters.redundancy == NONE) {
			error("Not implemented");
		}

		if (parameters.redundancy == DMR) {
			error("Not implemented");
		}

		if (parameters.redundancy == DMRMIXED) {
			error("Cannot execute DMRMIXED mode on half");
		}

	}

	if (parameters.precision == SINGLE) {
		if (parameters.redundancy == NONE) {
			error("Not implemented");
		}

		if (parameters.redundancy == DMR) {
			error("Not implemented");
		}

		if (parameters.redundancy == DMRMIXED) {
			error("Not implemented");
		}
	}

	if (parameters.precision == DOUBLE) {
		if (parameters.redundancy == NONE) {
			setup_double_unhardened(parameters, log);
		}

		if (parameters.redundancy == DMR) {
			setup_double_dmr_full(parameters, log);
		}

		if (parameters.redundancy == DMRMIXED) {
			setup_double_dmr_mixed(parameters, log);
		}
	}
}

//=============================================================================
//	MAIN FUNCTION
//=============================================================================

int main(int argc, char *argv[]) {
	std::cout << std::boolalpha;
	//=====================================================================
	//	CPU/MCPU VARIABLES
	//=====================================================================
	Parameters parameters(argc, argv);
	Log log;

	std::cout << parameters << std::endl;
	auto test_precision_description = "float";

	std::string test_info = std::string("type:") + test_precision_description
			+ "-precision streams:" + std::to_string(parameters.nstreams)
			+ " boxes:" + std::to_string(parameters.boxes) + " block_size:"
			+ std::to_string(NUMBER_THREADS);

	std::string test_name = std::string("cuda_") + test_precision_description
			+ "_lava";
	std::cout << "=================================" << std::endl;
	std::cout << test_precision_description << " " << test_name << std::endl;
	std::cout << "=================================" << std::endl;

	// timer
	if (!parameters.generate) {
		log = Log(test_name, test_info);
	}

	log.set_max_errors_iter(
	MAX_LOGGED_ERRORS_PER_STREAM * parameters.nstreams + 32);

#ifdef BUILDPROFILER

	std::string log_file_name = log.get_log_file_name();
	if(parameters.generate) {
		log_file_name = "/tmp/generate.log";
	}

	std::shared_ptr<rad::Profiler> profiler_thread = std::make_shared<rad::OBJTYPE>(0, log_file_name);

//START PROFILER THREAD
	profiler_thread->start_profile();
#endif

	/**
	 * Do the magic here
	 */
	choose_setup_option(parameters, log);

#ifdef BUILDPROFILER
	profiler_thread->end_profile();
#endif

	return 0;
}
