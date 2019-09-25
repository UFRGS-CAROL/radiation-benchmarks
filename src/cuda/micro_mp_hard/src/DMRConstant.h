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

template<const uint32 CHECK_BLOCK, typename half_t, typename real_t>
struct DMRConstant: public Microbenchmark<CHECK_BLOCK, half_t, real_t> {

	rad::DeviceVector<half_t> output_dev_1_lower, output_dev_2_lower,
			output_dev_3_lower;
	std::vector<half_t> output_host_1_lower, output_host_2_lower,
			output_host_3_lower;

	uint32 threshold_diff;

	virtual ~DMRConstant() = default;

	DMRConstant(const Parameters& parameters, Log& log) :
			Microbenchmark<CHECK_BLOCK, half_t, real_t>(parameters, log), threshold_diff(
					0) {
		auto r_size = this->parameters_.r_size;
		this->output_dev_1_lower.resize(r_size);
		this->output_dev_2_lower.resize(r_size);
		this->output_dev_3_lower.resize(r_size);
	}

	void call_kernel() {
		void (*kernel)(real_t* output_real_t_1, real_t* output_real_t_2,
				real_t* output_real_t_3, half_t* output_half_t_1,
				half_t* output_half_t_2, half_t* output_half_t_3);

		switch (this->parameters_.micro) {
		case ADD: {
			switch (this->parameters_.operation_num) {
			case 1: {
				kernel = &microbenchmark_kernel_add<ADD_UINT32_THRESHOLD_1,
						CHECK_BLOCK>;
				this->threshold_diff = ADD_UINT32_THRESHOLD_1;
				break;
			}
			case 10: {
				kernel = &microbenchmark_kernel_add<ADD_UINT32_THRESHOLD_10,
						CHECK_BLOCK>;
				this->threshold_diff = ADD_UINT32_THRESHOLD_10;
				break;
			}
			case 100: {
				kernel = &microbenchmark_kernel_add<ADD_UINT32_THRESHOLD_100,
						CHECK_BLOCK>;

				this->threshold_diff = ADD_UINT32_THRESHOLD_100;
				break;
			}
			case 1000: {
				kernel = &microbenchmark_kernel_add<ADD_UINT32_THRESHOLD_1000,
						CHECK_BLOCK>;

				this->threshold_diff = ADD_UINT32_THRESHOLD_1000;
				break;
			}

			case OPS: {
				kernel = &microbenchmark_kernel_add<ADD_UINT32_THRESHOLD_100000,
						CHECK_BLOCK>;
				this->threshold_diff = ADD_UINT32_THRESHOLD_100000;
				break;
			}
			}

			break;
		}

		case MUL: {
			switch (this->parameters_.operation_num) {
			case 1: {
				kernel = &microbenchmark_kernel_mul<MUL_UINT32_THRESHOLD_1,
						CHECK_BLOCK>;

				this->threshold_diff = MUL_UINT32_THRESHOLD_1;
				break;
			}
			case 10: {
				kernel = &microbenchmark_kernel_mul<MUL_UINT32_THRESHOLD_10,
						CHECK_BLOCK>;
				this->threshold_diff = MUL_UINT32_THRESHOLD_10;
				break;
			}
			case 100: {
				kernel = &microbenchmark_kernel_mul<MUL_UINT32_THRESHOLD_100,
						CHECK_BLOCK>;
				this->threshold_diff = MUL_UINT32_THRESHOLD_100;
				break;
			}

			case 1000: {
				kernel = &microbenchmark_kernel_mul<MUL_UINT32_THRESHOLD_1000,
						CHECK_BLOCK>;
				this->threshold_diff = MUL_UINT32_THRESHOLD_1000;
				break;
			}

			case OPS: {
				kernel = &microbenchmark_kernel_mul<MUL_UINT32_THRESHOLD_100000,
						CHECK_BLOCK>;
				this->threshold_diff = MUL_UINT32_THRESHOLD_100000;
				break;
			}
			}
			break;
		}

		case FMA: {
			switch (this->parameters_.operation_num) {
			case 1: {
				kernel = &microbenchmark_kernel_fma<FMA_UINT32_THRESHOLD_1,
						CHECK_BLOCK>;
				this->threshold_diff = FMA_UINT32_THRESHOLD_1;
				break;
			}
			case 10: {
				kernel = &microbenchmark_kernel_fma<FMA_UINT32_THRESHOLD_10,
						CHECK_BLOCK>;
				this->threshold_diff = FMA_UINT32_THRESHOLD_10;
				break;
			}
			case 100: {
				kernel = &microbenchmark_kernel_fma<FMA_UINT32_THRESHOLD_100,
						CHECK_BLOCK>;
				this->threshold_diff = FMA_UINT32_THRESHOLD_100;

				break;
			}
			case 1000: {
				kernel = &microbenchmark_kernel_fma<FMA_UINT32_THRESHOLD_1000,
						CHECK_BLOCK>;
				this->threshold_diff = FMA_UINT32_THRESHOLD_1000;
			}

			case OPS: {
				kernel = &microbenchmark_kernel_fma<FMA_UINT32_THRESHOLD_100000,
						CHECK_BLOCK>;
				this->threshold_diff = FMA_UINT32_THRESHOLD_100000;
				break;
			}
			}

			break;
		}
		}

		kernel<<<this->parameters_.grid_size, this->parameters_.block_size>>>(
				this->output_dev_1.data(), this->output_dev_2.data(),
				this->output_dev_3.data(), this->output_dev_1_lower.data(),
				this->output_dev_2_lower.data(),
				this->output_dev_3_lower.data());
	}

	inline half_t check_with_lower_precision(const uint64& i,
			uint64& memory_errors) override {
		auto val_output_1 = this->output_host_1_lower[i];
		auto val_output_2 = this->output_host_2_lower[i];
		auto val_output_3 = this->output_host_3_lower[i];
		auto val_voted = val_output_1;

		if ((val_output_1 != val_output_2) || (val_output_1 != val_output_3)) {
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

			//all three diverge
			if ((val_output_1 != val_output_2) && (val_output_2 != val_output_3)
					&& (val_output_1 != val_output_3)) {
#pragma omp critical
				{
					memory_errors++;
				}
			}
		}

		return val_voted;
	}

	bool cmp(double& lhs, double& rhs) {
		real_t diff = std::fabs(real_t(lhs) - rhs);
		if (diff > ZERO_DOUBLE) {
			return true;
		}
		return false;
	}

	bool cmp(float& lhs, double& rhs) {
		float rhs_ = float(rhs);
		uint32 lhs_data = reinterpret_cast<uint32&>(lhs);
		uint32 rhs_data = reinterpret_cast<uint32&>(rhs_);
		uint32 sub_res;
		if (lhs_data > rhs_data) {
			sub_res = lhs_data - rhs_data;
		} else {
			sub_res = rhs_data - lhs_data;
		}

		if (sub_res > this->threshold_diff) {
			return true;
		}

		return false;
	}

	uint64 copy_data_back() {
		this->output_host_1 = this->output_dev_1.to_vector();
		this->output_host_2 = this->output_dev_2.to_vector();
		this->output_host_3 = this->output_dev_3.to_vector();

		this->output_host_1_lower = this->output_dev_1_lower.to_vector();
		this->output_host_2_lower = this->output_dev_2_lower.to_vector();
		this->output_host_3_lower = this->output_dev_3_lower.to_vector();

		uint64 dmr_errors = 0;
		rad::checkFrameworkErrors(
				cudaMemcpyFromSymbol(&dmr_errors, errors, sizeof(uint64), 0,
						cudaMemcpyDeviceToHost));

		if (dmr_errors != 0) {
			std::string error_detail = "detected_dmr_errors: "
					+ std::to_string(dmr_errors);
			if (this->parameters_.verbose)
				std::cout << error_detail << std::endl;
			this->log_.log_info_detail(error_detail);
			this->log_.update_infos(1);
		}
		return dmr_errors;
	}

	uint32 get_max_threshold() {
		uint32 max_diff = 0;
		uint64 memory_errors = 0;
		uint64 nan_count = 0;
		uint64 inf_count = 0;
		uint64 zero_count = 0;
		float t, z;

#pragma omp parallel for shared(memory_errors, max_diff, nan_count, inf_count, zero_count, t, z)
		for (uint32 i = 0; i < this->gold_vector.size(); i++) {
			auto bigger = this->gold_vector[i];
			auto smaller = this->check_with_lower_precision(i, memory_errors);

			const float rhs_float = float(bigger);
			const float lhs_float = float(smaller);
			const uint32* lhs_ptr = (uint32*) &lhs_float;
			const uint32* rhs_ptr = (uint32*) &rhs_float;
			const uint32 lhs_data = *lhs_ptr;
			const uint32 rhs_data = *rhs_ptr;
			const uint32 sub_res =
					(lhs_data > rhs_data) ?
							lhs_data - rhs_data : rhs_data - lhs_data;

#pragma omp critical
			{
				nan_count += std::isnan(smaller);
				inf_count += std::isinf(smaller);
				zero_count += (smaller == 0.0);
				max_diff = max(max_diff, sub_res);
				if (max_diff == sub_res) {
					t = rhs_float;
					z = lhs_float;
				}
			}
		}

		std::cout << "SMALL PRECISION " << zero_count << " " << nan_count << " "
				<< inf_count << std::endl;
		std::cout << t << " " << z << std::endl;
		return max_diff;
	}
};

#endif /* DMRCONSTANT_H_ */
