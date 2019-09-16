/*
 * DMRNonConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef DMRNONCONSTANT_H_
#define DMRNONCONSTANT_H_

#include "Microbenchmark.h"

template<uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct DMRNonConstant: public Microbenchmark<CHECK_BLOCK, half_t, real_t> {

};

#endif /* DMRNONCONSTANT_H_ */
