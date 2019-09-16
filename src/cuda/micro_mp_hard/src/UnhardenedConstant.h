/*
 * UnhardenedConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef UNHARDENEDCONSTANT_H_
#define UNHARDENEDCONSTANT_H_

#include "Microbenchmark.h"

template<uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct UnhardenedConstant: public Microbenchmark<CHECK_BLOCK, half_t, real_t> {

};

#endif /* UNHARDENEDCONSTANT_H_ */
