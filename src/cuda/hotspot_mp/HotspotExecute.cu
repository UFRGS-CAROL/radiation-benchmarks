/*
 * HotspotExecute.cpp
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#include <cuda_fp16.h>


#include "HotspotExecute.h"
#include "kernels.h"

#ifdef LOGS
#include "log_helper.h"
#endif

#include "device_vector.h"


unsigned long long copy_errors() {
	unsigned long long errors_host = 0;
	//Copy errors first
	rad::checkFrameworkErrors(
			cudaMemcpyFromSymbol((void*)&errors_host, errors,
					sizeof(unsigned long long), 0));

	unsigned long long temp = 0;
	//Reset the errors variable
	rad::checkFrameworkErrors(
				cudaMemcpyToSymbol(errors, (void*)&temp,
						sizeof(unsigned long long), 0));
	return errors_host;
}


HotspotExecute::HotspotExecute(Parameters& setup_parameters, Log& log) :
		setup_params(setup_parameters), log(log), flops(0) {
	if (this->setup_params.verbose)
		std::cout << "WG size of kernel = " << BLOCK_SIZE << " x " << BLOCK_SIZE
				<< std::endl;
}

template<typename full, typename incomplete>
int HotspotExecute::compute_tran_temp(rad::DeviceVector<full>& power_array,
		rad::DeviceVector<full>& temp_array_input,
		rad::DeviceVector<full>& temp_array_output,
		rad::DeviceVector<incomplete>& temp_array_output_incomplete, int col,
		int row, int sim_time, int num_iterations, int blockCols, int blockRows,
		int borderCols, int borderRows, cudaStream_t stream) {
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

	int src = 1, dst = 0;
	full* MatrixPower = power_array.data();
	full* MatrixTemp[2] = { temp_array_input.data(), temp_array_output.data() };
	incomplete* MatrixTempIncomplete = temp_array_output_incomplete.data();
	if (this->setup_params.redundancy == NONE) {
		for (int t = 0; t < sim_time; t += num_iterations) {
			std::swap(src, dst);
			calculate_temp<full> <<<dimGrid, dimBlock, 0, stream>>>(
					MIN(num_iterations, sim_time - t), MatrixPower,
					MatrixTemp[src], MatrixTemp[dst], col, row, borderCols,
					borderRows, Cap_, Rx_, Ry_, Rz_, step_, time_elapsed);
			this->flops += col * row * MIN(num_iterations, sim_time - t) * 15;
		}
	} else {
		for (int t = 0; t < sim_time; t += num_iterations) {
			std::swap(src, dst);
			calculate_temp<full, incomplete> <<<dimGrid, dimBlock, 0, stream>>>(
					MIN(num_iterations, sim_time - t), MatrixPower,
					MatrixTemp[src], MatrixTemp[dst], MatrixTempIncomplete, col,
					row, borderCols, borderRows, Cap_, Rx_, Ry_, Rz_, step_,
					time_elapsed);
			this->flops += col * row * MIN(num_iterations, sim_time - t) * 15;
		}
	}

//	cudaStreamSynchronize(stream);
	return dst;
}

template<typename full, typename incomplete>
void HotspotExecute::generic_execute(int blockCols, int blockRows,
		int borderCols, int borderRows) {
	DataManagement<full, incomplete> hotspot_data(this->setup_params,
			this->log);
	hotspot_data.read_input();

	//====================================
	// Verbose in csv format
	if (this->setup_params.verbose == false) {
		std::cout << "output/s,iteration,time,output errors,relative errors"
				<< std::endl;
	}

	// ====================== MAIN BENCHMARK CYCLE ======================
	for (int loop = 0; loop < this->setup_params.setup_loops; loop++) {
		if (this->setup_params.verbose)
			std::cout << "======== Iteration #" << loop << " ========"
					<< std::endl;
		double global_time = this->log.mysecond();

		// ============ PREPARE ============
		double reload_time = this->log.mysecond();
		hotspot_data.reload();
		reload_time = this->log.mysecond();

		// ============ COMPUTE ============
		this->log.start_iteration_app();
		this->flops = 0;
		for (int streamIdx = 0; streamIdx < (this->setup_params.nstreams);
				streamIdx++) {
			rad::DeviceVector<full>& power_array_stream =
					hotspot_data.matrix_power_device[streamIdx];
			rad::DeviceVector<full>& temp_array_input_stream =
					hotspot_data.matrix_temperature_input_device[streamIdx];
			rad::DeviceVector<full>& temp_array_output_stream =
					hotspot_data.matrix_temperature_output_device[streamIdx];

			rad::DeviceVector<incomplete>& output_incomplete =
					hotspot_data.matrix_temperature_output_incomplete_device[streamIdx];

			hotspot_data.output_index[streamIdx] = compute_tran_temp<full,
					incomplete>(power_array_stream, temp_array_input_stream,
					temp_array_output_stream, output_incomplete, this->setup_params.grid_cols,
					this->setup_params.grid_rows, this->setup_params.sim_time,
					this->setup_params.pyramid_height, blockCols, blockRows,
					borderCols, borderRows, hotspot_data.streams[streamIdx]);
		}

		hotspot_data.sync();

		this->log.end_iteration_app();

		// ============ VALIDATE OUTPUT ============
		double copy_and_check_time = this->log.mysecond();

		hotspot_data.copy_from_gpu();
		hotspot_data.check_output_errors();

		auto dmr_errors = copy_errors();


		copy_and_check_time = this->log.mysecond();

		if (this->setup_params.verbose) {
			std::cout << "GPU prepare time: " << reload_time << "s"
					<< std::endl;

			// ============ MEASURE PERFORMANCE ============
			double outputpersec = double(
					((this->setup_params.grid_rows
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

			std::cout << "Gold check time: " << copy_and_check_time
					<< std::endl;

			//computing if the overall time is enough
			double iteration_time = this->log.mysecond() - global_time;

			std::cout << "Iteration time: " << iteration_time << " ("
					<< (this->log.iteration_time() / iteration_time) * 100.0
					<< "% Device)" << std::endl;
			std::cout << "Iteration errors " << this->log.error_count
					<< " DMR errors " << dmr_errors << std::endl;

			std::cout << "==============================" << std::endl;

		} else {
			//			std::cout << ".";
			// CSV format
			double outputpersec = double(
					((this->setup_params.grid_rows
							* this->setup_params.grid_rows
							* this->setup_params.nstreams)
							/ this->log.iteration_time()));
			std::cout << outputpersec << ",";
			std::cout << loop << ",";
			std::cout << this->log.iteration_time() << ",";
			std::cout << this->log.error_count << ",";
			std::cout << dmr_errors << std::endl;
		}

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

	switch (this->setup_params.redundancy) {
	case NONE:
	case DMR:
		switch (this->setup_params.precision) {
		case HALF:

			generic_execute<half, half>(blockCols, blockRows, borderCols,
					borderRows);
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
		break;

	case DMRMIXED:
		switch (this->setup_params.precision) {
		case SINGLE:
			generic_execute<float, half>(blockCols, blockRows, borderCols,
					borderRows);
			break;

		case DOUBLE:
			generic_execute<double, float>(blockCols, blockRows, borderCols,
					borderRows);
			break;

		}
		break;

	}

}
