//============================================================================
//	UPDATE
//============================================================================
//	14 APR 2011 Lukasz G. Szafaryn
//  2014-2018 Caio Lunardi
//  2018 Fernando Fernandes dos Santos
//  2019 Fernando Fernandes dos Santos new update

//=============================================================================
//	MAIN FUNCTION
//=============================================================================

#include "Parameters.h"
#include "Log.h"
#include "LavaExecute.h"


int main(int argc, char *argv[]) {
	Parameters setup_parameters(argc, argv);

	std::string app_name = "cuda_";
	app_name += setup_parameters.test_precision_description;

	std::string test_info = "type:";
	test_info += setup_parameters.test_precision_description;
	test_info += " streams:" + std::to_string(setup_parameters.nstreams);
	test_info += " boxes:" + std::to_string(setup_parameters.boxes);
	test_info += " block_size:" + std::to_string(NUMBER_THREADS);
	test_info += " redundancy:" + setup_parameters.test_redundancy_description;

	Log setup_log(app_name, test_info, !setup_parameters.generate);

	LavaExecute lava_execute(setup_parameters, setup_log);

	lava_execute.execute();

	return 0;
}

