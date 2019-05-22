/*
 * DataManagement.h
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#ifndef DATAMANAGEMENT_H_
#define DATAMANAGEMENT_H_

#include "DeviceVector.h"
#include "cuda_utils.h"
#include <fstream>      // std::fstream
#include <cmath> // isnan
#include <algorithm> // count

#include <omp.h>

template<typename full>
struct DataManagement {
	std::vector<std::vector<full>> matrix_temperature_input_host;
	std::vector<std::vector<full>> matrix_temperature_output_host;
	std::vector<std::vector<full>> matrix_power_host;

	std::vector<full> gold_temperature;
	std::vector<cudaStream_t> streams;

	std::vector<int> output_index;

	const Parameters& parameters;
	Log& log;

	// Alloc for multiple streams
	std::vector<DeviceVector<full> > matrix_temperature_input_device;
	std::vector<DeviceVector<full> > matrix_temperature_output_device;
	std::vector<DeviceVector<full> > matrix_power_device;

	std::vector<std::vector<full>> out_vector;

	DataManagement(Parameters& parameters, Log& log) :
			parameters(parameters), log(log) {

		this->output_index = std::vector<int>(this->parameters.nstreams);
		this->out_vector = std::vector<std::vector<full>>(this->parameters.nstreams);

		this->matrix_power_device = std::vector<DeviceVector<full>>(
				this->parameters.nstreams);
		this->matrix_temperature_input_device = std::vector<DeviceVector<full>>(
				this->parameters.nstreams);
		this->matrix_temperature_output_device =
				std::vector<DeviceVector<full>>(this->parameters.nstreams);

		this->matrix_power_host = std::vector<std::vector<full>>(
				this->parameters.nstreams);
		this->matrix_temperature_input_host = std::vector<std::vector<full>>(
				this->parameters.nstreams);
		this->matrix_temperature_output_host = std::vector<std::vector<full>>(
				this->parameters.nstreams);

		this->streams = std::vector<cudaStream_t>(this->parameters.nstreams);
		for (int stream = 0; stream < this->parameters.nstreams; stream++) {

			this->matrix_power_host[stream] = std::vector<full>(
					this->parameters.size);
			this->matrix_temperature_input_host[stream] = std::vector<full>(
					this->parameters.size);
			this->matrix_temperature_output_host[stream] = std::vector<full>(
					this->parameters.size);

			this->matrix_power_device[stream] = this->matrix_power_host[stream];
			this->matrix_temperature_input_device[stream] =
					this->matrix_temperature_input_host[stream];
			this->matrix_temperature_output_device[stream] =
					this->matrix_temperature_output_host[stream];

			checkFrameworkErrors(
					cudaStreamCreateWithFlags(&this->streams[stream],
							cudaStreamNonBlocking));

			this->out_vector[stream] = std::vector<full>(this->parameters.size);

		}

		this->gold_temperature = std::vector<full>(this->parameters.size);

	}

	void copy_from_gpu() {
		for (int stream = 0; stream < this->streams.size(); stream++) {
			DeviceVector<full>* output[2] = {
					&this->matrix_temperature_input_device[stream],
					&this->matrix_temperature_output_device[stream] };

			this->out_vector[stream] =
					output[this->output_index[stream]]->to_vector();
		}
	}

	void reload() {
		for (int stream = 0; stream < this->streams.size(); stream++) {
			this->matrix_power_device[stream] = this->matrix_power_host[stream];
			this->matrix_temperature_input_device[stream] =
					this->matrix_temperature_input_host[stream];
			this->matrix_temperature_output_device[stream].clear();
		}
	}

	void sync(){
		for (auto stream : this->streams) {
			checkFrameworkErrors(cudaStreamSynchronize(stream));
		}
		checkFrameworkErrors(cudaDeviceSynchronize());
		checkFrameworkErrors(cudaPeekAtLastError());
	}

	virtual ~DataManagement() {
		//Only destroy stream, the others will be automatically destroyed
		for (auto stream : this->streams) {
			checkFrameworkErrors(cudaStreamDestroy(stream));
		}
	}

	// Returns true if no errors are found. False if otherwise.
	void check_output_errors() {
		if (this->parameters.generate == true) {
			return;
		}

		size_t& host_errors = this->log.error_count;
		for (int stream = 0; stream < this->parameters.nstreams; stream++) {

#pragma omp parallel for shared(host_errors)
			for (int i = 0; i < this->parameters.grid_rows; i++) {
				for (int j = 0; j < this->parameters.grid_cols; j++) {
					int index = i * this->parameters.grid_rows + j;

					double valGold = this->gold_temperature[index];
					double valOutput = this->out_vector[stream][index];

					if (valGold != valOutput) {
#pragma omp critical
						{
							char error_detail[150];
							snprintf(error_detail, 150,
									"stream: %d, p: [%d, %d], r: %1.20e, e: %1.20e",
									stream, i, j, valOutput, valGold);
							this->log.log_error(std::string(error_detail));
							if (this->parameters.verbose && (host_errors < 10))
								std::cout << error_detail << std::endl;

						}
					}
				}
			}
		}

		if (host_errors != 0) {
			if (!this->parameters.verbose)
				std::cout << "#";
			else
				std::cout << "Output errors: " << host_errors << std::endl;
		}
		this->log.update_error_count();
	}

	void read_input() {
		double timestamp = Log::mysecond();
		// =================== Read all files
		std::fstream temp_file, power_file, gold_file;
		temp_file.open(this->parameters.tfile, std::fstream::in);
		power_file.open(this->parameters.pfile, std::fstream::in);
		gold_file.open(this->parameters.ofile,
				std::fstream::in | std::fstream::binary);

		int num_zeros = 0;
		int num_nans = 0;

		if (!temp_file.is_open()) {
			std::cerr << "The temp file was not opened" << std::endl;
			exit(EXIT_FAILURE);
		}

		if (!power_file.is_open()) {
			std::cerr << "The power file was not opened" << std::endl;
			exit(EXIT_FAILURE);
		}

		if (this->parameters.generate == false) {
			if (gold_file.is_open() == false) {
				std::cerr << "The gold file was not opened" << std::endl;
				exit(EXIT_FAILURE);
			}
			// reading from gold
			gold_file.read((char*) this->gold_temperature.data(),
					sizeof(full) * this->parameters.size);
		}

		std::vector<full> temperature(this->parameters.size);
		std::vector<full> power(this->parameters.size);

		for (int i = 0; i < this->parameters.grid_rows; i++) {
			for (int j = 0; j < this->parameters.grid_cols; j++) {
				if (temp_file.eof()) {
					std::cerr << "[" << i << "," << j << "] size: "
							<< this->parameters.size << std::endl;
					std::cerr << "not enough lines in temp file" << std::endl;
					exit(EXIT_FAILURE);
				}

				float temp_val;
				temp_file >> temp_val;

				temperature[i * this->parameters.grid_cols + j] = full(
						temp_val);

				if (temp_val == 0)
					num_zeros++;
				if (std::isnan(temp_val))
					num_nans++;

				float power_val;
				if (power_file.eof()) {
					std::cerr << "[" << i << "," << j << "] size: "
							<< this->parameters.size << std::endl;
					std::cerr << "not enough lines in power file" << std::endl;
					exit(EXIT_FAILURE);
				}

				power_file >> power_val;

				power[i * this->parameters.grid_cols + j] = full(power_val);

				if (power_val == 0)
					num_zeros++;
				if (std::isnan(power_val))
					num_nans++;
			}
		}

		std::cout << "Zeros in the input: " << num_zeros << std::endl;
		std::cout << "NaNs in the input: " << num_nans << std::endl;

		// =================== FAULT INJECTION
		if (this->parameters.fault_injection) {
			temperature[32] = 6.231235;
			std::cout << "!!!!!!!!! Injected error: temperature[32] = "
					<< float(temperature[32]) << std::endl;
		}

		for (int stream = 0; stream < this->streams.size(); stream++) {
			this->matrix_power_host[stream] = power;
			this->matrix_temperature_input_host[stream] = temperature;
		}
		// ==================================

		power_file.close();
		temp_file.close();
		if (this->parameters.generate)
			gold_file.close();

		if (this->parameters.verbose)
			std::cout << "readInput time: " << Log::mysecond() - timestamp
					<< std::endl;
	}

	void write_output() {
		if (this->parameters.generate == false) {
			return;
		}

		// =================== Write output to gold file
		std::fstream gold_file(this->parameters.ofile,
				std::fstream::out | std::fstream::binary);

		if (!gold_file.is_open()) {
			std::cerr << "The gold file was not opened" << std::endl;
			exit(EXIT_FAILURE);
		}

		gold_file.write((char*)  this->out_vector[0].data(),
				sizeof(full) * this->parameters.size);

		gold_file.close();

		int nan = 0;
		int zero = 0;
		for (auto n :  this->out_vector[0]) {
			if (std::isnan(float(n)))
				nan++;
			if (float(n) == 0)
				zero++;
		}
		std::cout << "Gold Zeros in the output: " << zero << std::endl;
		std::cout << "Gold NaNs in the output: " << nan << std::endl;
	}
};

#endif /* DATAMANAGEMENT_H_ */
