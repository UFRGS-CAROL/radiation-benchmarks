/*
 * kernels.h
 *
 *  Created on: 26/01/2019
 *      Author: fernando
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <vector>
#include <string>
#include <memory>
#include "Log.h"
#include "utils.h"
#include "CacheLine.h"


#define BLOCK_SIZE 32

enum Board {
	K40, TEGRAX2, TITANV, BOARD_COUNT,
};

struct Parameters {
	uint32 number_of_sms;
	Board device;
	uint32 shared_memory_size;
	uint32 l2_size;
	uint64 one_second_cycles; // the necessary number of cycles to count one second
	Log *log;
	std::string board_name;
	byte t_byte;

	//register file size
	uint32 registers_per_block;

	//const memory
	uint32 const_memory_per_block;

};


struct Tuple {
	std::vector<int32> misses;
	std::vector<int32> hits;
	std::vector<byte> cache_lines;
	std::vector<uint32> register_file;
	uint64 errors;

};

Tuple test_l1_cache(const Parameters&);
Tuple test_l2_cache(const Parameters&);
Tuple test_shared_memory(const Parameters&);
Tuple test_read_only_cache(const Parameters&);
Tuple test_register_file(const Parameters&);

#endif /* KERNELS_H_ */
