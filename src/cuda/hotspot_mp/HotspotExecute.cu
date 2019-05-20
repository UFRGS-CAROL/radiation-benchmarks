/*
 * HotspotExecute.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include "HotspotExecute.h"
#include "none_kernels.h"
#include "half.hpp"

#include <cuda_fp16.h>

#ifdef LOGS
#include "log_helper.h"
#endif
// The timestamp is updated on every log_helper function call.

HotspotExecute::HotspotExecute(Parameters& setup_parameters, Log& log) :
		setup_params(setup_parameters), log(log) {
}

template<typename full, typename incomplete>
int HotspotExecute::compute_tran_temp(DeviceVector<full>& power_array,
		DeviceVector<full>& temp_array_input,
		DeviceVector<full>& temp_array_output, int col, int row, int sim_time,
		int num_iterations, int blockCols, int blockRows, int borderCols,
		int borderRows, cudaStream_t stream, double& flops) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);

	incomplete t_chip(0.0005);
	incomplete chip_height(0.016);
	incomplete chip_width(0.016);

	incomplete grid_height = chip_height / incomplete(row);
	incomplete grid_width = chip_width / incomplete(col);

	full Cap =
			BIGGEST_TYPE_CAST(
					incomplete(FACTOR_CHIP) * incomplete(SPEC_HEAT_SI) * t_chip * grid_width * grid_height);
	full Rx =
			BIGGEST_TYPE_CAST(
					grid_width / (incomplete(2.0) * incomplete(K_SI) * t_chip * grid_height));
	full Ry =
			BIGGEST_TYPE_CAST(
					grid_height / (incomplete(2.0) * incomplete(K_SI) * t_chip * grid_width));
	full Rz = BIGGEST_TYPE_CAST(
			t_chip / (incomplete(K_SI) * grid_height * grid_width));

	incomplete max_slope = incomplete(MAX_PD)
			/ (incomplete(FACTOR_CHIP) * t_chip * incomplete(SPEC_HEAT_SI));
	full step = BIGGEST_TYPE_CAST(incomplete(PRECISION) / max_slope);
	full time_elapsed = (0.001);

	std::cout << BIGGEST_TYPE_CAST(Cap) << " " << BIGGEST_TYPE_CAST(Rx) << " "
			<< BIGGEST_TYPE_CAST(Ry) << " " << BIGGEST_TYPE_CAST(Rz) << "\n";

	int src = 0, dst = 1;
	full* MatrixPower = power_array.data;
	full* MatrixTemp[2] = { temp_array_input.data, temp_array_output.data };

	for (int t = 0; t < sim_time; t += num_iterations) {
		calculate_temp<full> <<<dimGrid, dimBlock, 0, stream>>>(
				MIN(num_iterations, sim_time - t), MatrixPower, MatrixTemp[src],
				MatrixTemp[dst], col, row, borderCols, borderRows, Cap, Rx, Ry,
				Rz, step, time_elapsed);
		flops += col * row * MIN(num_iterations, sim_time - t) * 15;
		std::swap(src, dst);
	}
//	cudaStreamSynchronize(stream);
	return dst;
}

template<typename full, typename incomplete>
void HotspotExecute::generic_execute(int blockCols, int blockRows,
		int borderCols, int borderRows) {
	DataManagement<full> hotspot_data(this->setup_params);
	hotspot_data.read_input();
	// ====================== MAIN BENCHMARK CYCLE ======================
	for (int loop = 0; loop < this->setup_params.setup_loops; loop++) {
		if (this->setup_params.verbose)
			std::cout << "======== Iteration #" << loop << " ========"
					<< std::endl;

		double globaltime = this->log.mysecond();
		// ============ PREPARE ============
		std::vector<int> ret(this->setup_params.nstreams);
		double timestamp = this->log.mysecond();
		hotspot_data.reload();
		if (this->setup_params.verbose)
			std::cout << "GPU prepare time: "
					<< this->log.mysecond() - timestamp << "s" << std::endl;

		// ============ COMPUTE ============
		double kernel_time = this->log.mysecond();
		this->log.start_iteration_app();
		double flops = 0;
		for (int streamIdx = 0; streamIdx < (this->setup_params.nstreams);
				streamIdx++) {
			DeviceVector<full>& power_array_stream =
					hotspot_data.matrix_power_device[streamIdx];
			DeviceVector<full>& temp_array_input_stream =
					hotspot_data.matrix_temperature_input_device[streamIdx];
			DeviceVector<full>& temp_array_output_stream =
					hotspot_data.matrix_temperature_output_device[streamIdx];

			ret[streamIdx] = compute_tran_temp<full, incomplete>(
					power_array_stream, temp_array_input_stream,
					temp_array_output_stream, this->setup_params.grid_cols,
					this->setup_params.grid_rows, this->setup_params.sim_time,
					this->setup_params.pyramid_height, blockCols, blockRows,
					borderCols, borderRows, hotspot_data.streams[streamIdx],
					flops);
		}

		for (auto stream : hotspot_data.streams) {
			checkFrameworkErrors(cudaStreamSynchronize(stream));
		}
		this->log.end_iteration_app();
		kernel_time = this->log.mysecond() - kernel_time;
		// ============ MEASURE PERFORMANCE ============
		if (this->setup_params.verbose) {
			double outputpersec =
					(double) (((this->setup_params.grid_rows
							* this->setup_params.grid_rows
							* this->setup_params.nstreams) / kernel_time));
			std::cout << "Kernel time: " << kernel_time << std::endl;
			std::cout << "Performance - SIZE:" << this->setup_params.grid_rows
					<< " OUTPUT/S: " << outputpersec << " FLOPS: "
					<< flops / kernel_time << " (GFLOPS: "
					<< flops / (kernel_time * 1e9) << ")" << std::endl;
		}
		// ============ VALIDATE OUTPUT ============
		timestamp = this->log.mysecond();
		int kernel_errors = 0;

		hotspot_data.copy_from_gpu();

		if (this->setup_params.generate) {
			hotspot_data.write_output();
		} else {
			hotspot_data.check_output_errors();
		}

		if (this->setup_params.verbose)
			std::cout << "Gold check time: " << this->log.mysecond() - timestamp
					<< std::endl;

		if ((kernel_errors != 0) && !(this->setup_params.verbose))
			std::cout << ".";

		double iteration_time = this->log.mysecond() - globaltime;
		if (this->setup_params.verbose)
			std::cout << "Iteration time: " << iteration_time << " ("
					<< (kernel_time / iteration_time) * 100.0 << "% Device)\n"
					<< std::endl;

		if (this->setup_params.verbose)
			std::cout << ("==============================\n");

	}
}

HotspotExecute::~HotspotExecute() {
}

void HotspotExecute::run() {
	// ===============  pyramid parameters
	int borderCols = (this->setup_params.pyramid_height) * EXPAND_RATE / 2;
	int borderRows = (this->setup_params.pyramid_height) * EXPAND_RATE / 2;
	int smallBlockCol = BLOCK_SIZE
			- (this->setup_params.pyramid_height) * EXPAND_RATE;
	int smallBlockRow = BLOCK_SIZE
			- (this->setup_params.pyramid_height) * EXPAND_RATE;
	int blockCols = this->setup_params.grid_cols / smallBlockCol
			+ ((this->setup_params.grid_cols % smallBlockCol == 0) ? 0 : 1);
	int blockRows = this->setup_params.grid_rows / smallBlockRow
			+ ((this->setup_params.grid_rows % smallBlockRow == 0) ? 0 : 1);

	this->setup_params.size = (this->setup_params.grid_cols)
			* (this->setup_params.grid_rows);

	switch (this->setup_params.precision) {
	case HALF:

		generic_execute<half, half_float::half>(blockCols, blockRows,
				borderCols, borderRows);
		break;

	case SINGLE:
		generic_execute<float, float>(blockCols, blockRows, borderCols,
				borderRows);
		break;

	case DOUBLE:
		generic_execute<double, double>(blockCols, blockRows, borderCols,
				borderRows);
		break;

	}

}
