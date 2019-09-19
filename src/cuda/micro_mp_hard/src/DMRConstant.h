/*
 * DMRConstant.h
 *
 *  Created on: 15/09/2019
 *      Author: fernando
 */

#ifndef DMRCONSTANT_H_
#define DMRCONSTANT_H_

#include "Microbenchmark.h"
#include "dmr_constant_kernels.h"
#include "common.h"
#include "Parameters.h"

template<uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct DMRConstant: public Microbenchmark<CHECK_BLOCK, half_t, real_t> {

	rad::DeviceVector<half_t> output_dev_1_lower, output_dev_2_lower,
			output_dev_3_lower;
	std::vector<half_t> output_host_1_lower, output_host_2_lower,
			output_host_3_lower;

	virtual ~DMRConstant() = default;

	DMRConstant(const Parameters& parameters, Log& log) :
			Microbenchmark<CHECK_BLOCK, half_t, real_t>(parameters, log) {
		auto r_size = this->parameters_.r_size;
		this->output_dev_1_lower.resize(r_size);
		this->output_dev_2_lower.resize(r_size);
		this->output_dev_3_lower.resize(r_size);
	}

	virtual void call_kernel() override {
		//================== Device computation
		switch (this->parameters_.micro) {
		case ADD:
			microbenchmark_kernel_add<CHECK_BLOCK> <<<
					this->parameters_.grid_size, this->parameters_.block_size>>>(
					this->output_dev_1.data(), this->output_dev_2.data(),
					this->output_dev_3.data(), this->output_dev_1_lower.data(),
					this->output_dev_2_lower.data(),
					this->output_dev_3_lower.data());
			break;
		case MUL:
			microbenchmark_kernel_mul<CHECK_BLOCK> <<<
					this->parameters_.grid_size, this->parameters_.block_size>>>(
					this->output_dev_1.data(), this->output_dev_2.data(),
					this->output_dev_3.data(), this->output_dev_1_lower.data(),
					this->output_dev_2_lower.data(),
					this->output_dev_3_lower.data());
			break;
		case FMA:
			microbenchmark_kernel_fma<CHECK_BLOCK> <<<
					this->parameters_.grid_size, this->parameters_.block_size>>>(
					this->output_dev_1.data(), this->output_dev_2.data(),
					this->output_dev_3.data(), this->output_dev_1_lower.data(),
					this->output_dev_2_lower.data(),
					this->output_dev_3_lower.data());
			break;
		}
	}

	virtual inline double check_with_lower_precision(const real_t& val,
			const uint64& i, uint64& memory_errors) override {
		auto val_output_1 = this->output_host_1_lower[i];
		auto val_output_2 = this->output_host_2_lower[i];
		auto val_output_3 = this->output_host_3_lower[i];
		auto val_voted = val_output_1;

		if ((val_output_1 != val_output_2) || (val_output_1 != val_output_3)) {
#pragma omp critical
			{
				memory_errors++;
			}

			if (val_output_2 == val_output_3) {
				// Only value 0 diverge
				val_voted = val_output_2;
			} else if (val_output_1 == val_output_3) {
				// Only value 1 diverge
				val_voted = val_output_1;
			} else if (val_output_1 == val_output_2) {
				// Only value 2 diverge
				val_voted = val_output_1;
			}
		}

		return double(val_voted);
	}

	virtual inline uint64 copy_data_back() override {
		Microbenchmark<CHECK_BLOCK, half_t, real_t>::copy_data_back();
		this->output_host_1_lower = this->output_dev_1_lower.to_vector();
		this->output_host_2_lower = this->output_dev_2_lower.to_vector();
		this->output_host_3_lower = this->output_dev_3_lower.to_vector();

		uint64 relative_errors = 0;
		rad::checkFrameworkErrors(
				cudaMemcpyFromSymbol(&relative_errors, errors, sizeof(uint64),
						0, cudaMemcpyDeviceToHost));

		return relative_errors;

	}

};

#endif /* DMRCONSTANT_H_ */
