/*
 * UnhardenedConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef UNHARDENEDCONSTANT_H_
#define UNHARDENEDCONSTANT_H_

#include "Microbenchmark.h"
#include "none_kernels.h"

template<uint32 CHECK_BLOCK, typename real_t>
struct UnhardenedConstant: public Microbenchmark<CHECK_BLOCK, real_t, real_t> {

	UnhardenedConstant(const Parameters& parameters, Log& log) :
			Microbenchmark<CHECK_BLOCK, real_t, real_t>(parameters, log) {
	}

	bool cmp(real_t& lhs, real_t& rhs) {
		return false;
	}

	real_t check_with_lower_precision(const uint64& i,
			uint64& memory_errors) {
		return 0.0;
	}

	uint32 get_max_threshold() {
		return 0;
	}

	uint64 copy_data_back() {
		this->output_host_1 = this->output_dev_1.to_vector();
		this->output_host_2 = this->output_dev_2.to_vector();
		this->output_host_3 = this->output_dev_3.to_vector();
		return 0;
	}

	void call_kernel() {
		//================== Device computation
		switch (this->parameters_.micro) {
		case ADD:
			microbenchmark_kernel_add<<<this->parameters_.grid_size,
					this->parameters_.block_size>>>(this->output_dev_1.data(),
					this->output_dev_2.data(), this->output_dev_3.data());
			break;
		case MUL:
			microbenchmark_kernel_mul<<<this->parameters_.grid_size,
					this->parameters_.block_size>>>(this->output_dev_1.data(),
					this->output_dev_2.data(), this->output_dev_3.data());
			break;
		case FMA:
			microbenchmark_kernel_fma<<<this->parameters_.grid_size,
					this->parameters_.block_size>>>(this->output_dev_1.data(),
					this->output_dev_2.data(), this->output_dev_3.data());
			break;
		}
	}
};

#endif /* UNHARDENEDCONSTANT_H_ */
