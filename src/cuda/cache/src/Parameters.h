/*
 * Parameters.h
 *
 *  Created on: Sep 3, 2019
 *      Author: carol
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "utils.h"
#include <unordered_map>

#define BLOCK_SIZE 32
#define MAX_THREAD_PER_BLOCK BLOCK_SIZE * BLOCK_SIZE

/**
 * It was 256 registers per threads when this program was created
 */
#if __CUDA_ARCH__ <= 750

//CACHE LINE PARAMETERS TO TEST RF, L1, and L2
#define RF_SIZE 256
#define CACHE_LINE_SIZE 128U
#define CACHE_LINE_SIZE_BY_INT64 (CACHE_LINE_SIZE / 8)

//SHARED MEMORY PARAMETERS TO FORCE THE BLOCKS EXECUTE
//WHITHIN A SM
#define MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1 2 * 1024
#define MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1 1 * 1024

//SHARED MEMORY PARAMETERS TO TEST SHARED MEMORY
#define MAX_VOLTA_SHARED_MEMORY 48 * 1024
#define MAX_KEPLER_SHARED_MEMORY 48 * 1024

//KEPLER L1 MEMORY PARAMETERS
#define MAX_VOLTA_L1_MEMORY 48 * 1024
#define MAX_KEPLER_L1_MEMORY 24 * 1024

//READ ONLY PARAMETERS FOR KEPLER
#define MAX_VOLTA_CONSTANT_MEMORY 64 * 1024
#define MAX_KEPLER_CONSTANT_MEMORY 64 * 1024


#define BLOCK_PER_SM 1
#endif

#define DEVICE_INDEX 0 //Radiation test can be done only one device at time

typedef enum {
	K20, K40, TEGRAX2, XAVIER, TITANV, BOARD_COUNT,
} Board;

struct Parameters {
	uint32 number_of_sms;
	Board device;
	uint32 shared_memory_size;
	uint32 l2_size;
	uint64 one_second_cycles; // the necessary number of cycles to count one second

	std::string board_name;

	//register file size
	uint32 registers_per_block;

	//const memory
	uint32 const_memory_per_block;

	uint64 clock_rate;

	std::unordered_map<std::string, Board> devices_name = {
	//Tesla K20
			{ "Tesla K20c", K20 },
			//Tesla K40
			{ "Tesla K40c", K40 },
			// Titan V
			{ "TITAN V", TITANV },
			//Xavier
			{ "Xavier", XAVIER }
	//Other
			};

	int32 iterations;
	bool verbose;
	std::string test_mode;
	uint64 errors;
	uint64 infos;
	uint32 seconds_sleep;
//	std::shared_ptr<rad::Log> log_ptr;

	Parameters(int argc, char** argv);

	friend std::ostream& operator<<(std::ostream& os, const Parameters& par);

	cudaDeviceProp get_device_information(int dev) ;

	void usage(char*);
};

#endif /* PARAMETERS_H_ */
