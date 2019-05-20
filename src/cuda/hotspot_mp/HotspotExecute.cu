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

template<typename full>
int HotspotExecute::compute_tran_temp(DeviceVector<full>& power_array,
		DeviceVector<full>& temp_array_input,
		DeviceVector<full>& temp_array_output, int col, int row, int sim_time,
		int num_iterations, int blockCols, int blockRows, int borderCols,
		int borderRows, cudaStream_t stream, double& flops) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(blockCols, blockRows);

	// Default values of hotpsot
	DefaultType t_chip(0.0005);
	DefaultType chip_height(0.016);
	DefaultType chip_width(0.016);
	DefaultType grid_height = chip_height / row;
	DefaultType grid_width = chip_width / col;
	DefaultType Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width
			* grid_height;
	DefaultType Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	DefaultType Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	DefaultType Rz = t_chip / (K_SI * grid_height * grid_width);
	DefaultType max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	DefaultType step = PRECISION / max_slope;

	//New precision converted types
	full time_elapsed = 0.001;
	full Cap_ = Cap;
	full Rx_ = Rx;
	full Ry_ = Ry;
	full Rz_ = Rz;
	full step_ = step;

	int src = 0, dst = 1;
	full* MatrixPower = power_array.data;
	full* MatrixTemp[2] = { temp_array_input.data, temp_array_output.data };

	std::cout << " output " << MatrixTemp[0] << " input " << MatrixTemp[1]
			<< std::endl;

	for (int t = 0; t < sim_time; t += num_iterations) {
		calculate_temp<full> <<<dimGrid, dimBlock, 0, stream>>>(
				MIN(num_iterations, sim_time - t), MatrixPower, MatrixTemp[src],
				MatrixTemp[dst], col, row, borderCols, borderRows, Cap_, Rx_,
				Ry_, Rz_, step_, time_elapsed);
		flops += col * row * MIN(num_iterations, sim_time - t) * 15;
		std::swap(src, dst);
	}
//	cudaStreamSynchronize(stream);
	return dst;
}

template<typename full>
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

//			ret[streamIdx] =
			compute_tran_temp<full>(power_array_stream, temp_array_input_stream,
					temp_array_output_stream, this->setup_params.grid_cols,
					this->setup_params.grid_rows, this->setup_params.sim_time,
					this->setup_params.pyramid_height, blockCols, blockRows,
					borderCols, borderRows, hotspot_data.streams[streamIdx],
					flops);
		}

		for (auto stream : hotspot_data.streams) {
			checkFrameworkErrors(cudaStreamSynchronize(stream));
		}
		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());

		this->log.end_iteration_app();
		// ============ MEASURE PERFORMANCE ============
		if (this->setup_params.verbose) {
			double outputpersec =
					(double) (((this->setup_params.grid_rows
							* this->setup_params.grid_rows
							* this->setup_params.nstreams)
							/ this->log.iteration_time()));
			std::cout << "Kernel time: " << this->log.iteration_time()
					<< std::endl;
			std::cout << "Performance - SIZE:" << this->setup_params.grid_rows
					<< " OUTPUT/S: " << outputpersec << " FLOPS: "
					<< flops / this->log.iteration_time() << " (GFLOPS: "
					<< flops / (this->log.iteration_time() * 1e9) << ")"
					<< std::endl;
		}
		// ============ VALIDATE OUTPUT ============
		timestamp = this->log.mysecond();
		int kernel_errors = 0;

		hotspot_data.copy_from_gpu();
		hotspot_data.check_output_errors();

		if (this->setup_params.verbose)
			std::cout << "Gold check time: " << this->log.mysecond() - timestamp
					<< std::endl;

		if ((kernel_errors != 0) && !(this->setup_params.verbose))
			std::cout << ".";

		if (this->setup_params.verbose) {
			//computing if the overall time is enough
			double iteration_time = this->log.mysecond() - globaltime;

			std::cout << "Iteration time: " << iteration_time << " ("
					<< (this->log.iteration_time() / iteration_time) * 100.0
					<< "% Device)\n" << std::endl;
		}
		if (this->setup_params.verbose)
			std::cout << ("==============================\n");

	}

	//this function already check if must generate a gold
	// or not
	hotspot_data.write_output();
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

		generic_execute<half>(blockCols, blockRows, borderCols, borderRows);
		break;

	case SINGLE:
		generic_execute<float>(blockCols, blockRows, borderCols, borderRows);
		break;

	case DOUBLE:
		generic_execute<double>(blockCols, blockRows, borderCols, borderRows);
		break;

	}

}
