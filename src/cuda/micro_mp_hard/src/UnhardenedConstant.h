/*
 * UnhardenedConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef UNHARDENEDCONSTANT_H_
#define UNHARDENEDCONSTANT_H_

#include "Microbenchmark.h"

template<uint32 CHECK_BLOCK, typename real_t>
struct UnhardenedConstant: public Microbenchmark<CHECK_BLOCK, real_t, real_t> {

	UnhardenedConstant(const Parameters& parameters, Log& log) : Microbenchmark<CHECK_BLOCK, real_t, real_t>(parameters, log){}
};

#endif /* UNHARDENEDCONSTANT_H_ */
