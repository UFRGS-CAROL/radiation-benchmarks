/*
 * DMRConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef DMRCONSTANT_H_
#define DMRCONSTANT_H_

#include "Microbenchmark.h"

template<uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct DMRConstant : public Microbenchmark<CHECK_BLOCK, half_t, real_t> {

};


#endif /* DMRCONSTANT_H_ */
