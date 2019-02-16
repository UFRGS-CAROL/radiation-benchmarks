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
	K20, K40, TEGRAX2, XAVIER, TITANV, BOARD_COUNT,
};

enum MEM {
		L1, L2, SHARED, RF
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
	std::vector<byte> cache_lines2;
	std::vector<byte> cache_lines3;
	
	std::vector<uint32> register_file;
	std::vector<uint32> register_file2;
	std::vector<uint32> register_file3;
	uint64 errors;
	uint64 errors2;
	uint64 errors3;

};



template<uint32 LINE_SIZE> 
std::vector<byte> move_to_byte(const std::vector<CacheLine<LINE_SIZE> >& T){
        std::vector<byte> ret(T.size() * LINE_SIZE);
#pragma unroll
        for(uint32 i = 0; i < T.size(); i++){
                for (uint32 j = 0; j < LINE_SIZE; j++) {
                        ret[i * LINE_SIZE + j] = T[i].t[j];
                }
        }
        return ret;        
}

Tuple test_l1_cache(const Parameters&);
Tuple test_l2_cache(const Parameters&);
Tuple test_shared_memory(const Parameters&);
Tuple test_read_only_cache(const Parameters&);
Tuple test_register_file(const Parameters&);

#endif /* KERNELS_H_ */
