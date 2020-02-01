/*
 * util.cpp
 *
 *  Created on: Feb 1, 2020
 *      Author: fernando
 */

#include "utils.h"


void __throw_line(std::string err, std::string line, std::string file){
	throw std::runtime_error("ERROR at " + file + ":" + line);

}
