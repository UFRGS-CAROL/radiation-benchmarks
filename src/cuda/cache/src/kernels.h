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
        
	std::vector<int32> misses = {};
	std::vector<int32> hits = {};
	std::vector<byte> cache_lines1;
	std::vector<byte> cache_lines2;
	std::vector<byte> cache_lines3;
	
	std::vector<uint32> register_file1;
	std::vector<uint32> register_file2;
	std::vector<uint32> register_file3;

        template<uint32 LINE_SIZE> 
        void move_to_byte(const std::vector<CacheLine<LINE_SIZE> >& T1, const std::vector<CacheLine<LINE_SIZE> >& T2, const std::vector<CacheLine<LINE_SIZE> >& T3){
                this->cache_lines1.resize(T1.size() * LINE_SIZE);
                this->cache_lines2.resize(T2.size() * LINE_SIZE);
                this->cache_lines3.resize(T3.size() * LINE_SIZE);
                
                #pragma unroll
                for(uint32 i = 0; i < T1.size(); i++){
                        for (uint32 j = 0; j < LINE_SIZE; j++) {
                                this->cache_lines1[i * LINE_SIZE + j] = T1[i][j];
                                this->cache_lines2[i * LINE_SIZE + j] = T2[i][j];
                                this->cache_lines3[i * LINE_SIZE + j] = T3[i][j];
                        }
                }
        }
        
        template<typename T = std::vector<uint32> >
        void move_register_file(T& rf1, T& rf2, T& rf3){
                this->register_file1 = rf1;
                this->register_file2 = rf2;
                this->register_file3 = rf3;
        }
        
        void set_misses(std::vector<int32>& miss){
                this->misses = miss;
        }
        
        void set_hits(std::vector<int32>& hits){
                this->hits = hits;
        }
        

};





Tuple test_l1_cache(const Parameters&);
Tuple test_l2_cache(const Parameters&);
Tuple test_shared_memory(const Parameters&);
//Tuple test_read_only_cache(const Parameters&);
Tuple test_register_file(const Parameters&);

#endif /* KERNELS_H_ */
