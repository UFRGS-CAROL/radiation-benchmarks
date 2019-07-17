/*
 * DataManagement.h
 *
 *  Created on: 18/05/2019
 *      Author: fernando
 */

#ifndef DATAMANAGEMENT_H_
#define DATAMANAGEMENT_H_

#include <fstream>      // std::fstream
#include <cmath> // isnan
#include <algorithm> // count
#include <omp.h>

#include "device_functions.h"
#include "cuda_utils.h"
#include "device_vector.h"

template<typename full, typename incomplete>
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
	std::vector<rad::DeviceVector<full> > matrix_temperature_input_device;
	std::vector<rad::DeviceVector<full> > matrix_temperature_output_device;
	std::vector<rad::DeviceVector<full> > matrix_power_device;

	std::vector<rad::DeviceVector<incomplete>> matrix_temperature_output_incomplete_device;
	std::vector<std::vector<incomplete>> matrix_temperature_output_incomplete_host;

	std::vector<std::vector<full>> out_vector;

	DataManagement(Parameters& parameters, Log& log) :
			parameters(parameters), log(log) {

		this->output_index = std::vector<int>(this->parameters.nstreams);
		this->out_vector = std::vector < std::vector
				< full >> (this->parameters.nstreams);

		this->matrix_power_device = std::vector < rad::DeviceVector
				< full >> (this->parameters.nstreams);
		this->matrix_temperature_input_device = std::vector < rad::DeviceVector
				< full >> (this->parameters.nstreams);
		this->matrix_temperature_output_device = std::vector < rad::DeviceVector
				< full >> (this->parameters.nstreams);

		this->matrix_power_host = std::vector < std::vector
				< full >> (this->parameters.nstreams);
		this->matrix_temperature_input_host = std::vector < std::vector
				< full >> (this->parameters.nstreams);
		this->matrix_temperature_output_host = std::vector < std::vector
				< full >> (this->parameters.nstreams);

		this->streams = std::vector < cudaStream_t
				> (this->parameters.nstreams);

		this->matrix_temperature_output_incomplete_host.resize(
				this->parameters.nstreams);
		this->matrix_temperature_output_incomplete_device.resize(
				this->parameters.nstreams);

		for (int stream = 0; stream < this->parameters.nstreams; stream++) {

			this->matrix_power_host[stream] = std::vector < full
					> (this->parameters.size);
			this->matrix_temperature_input_host[stream] = std::vector < full
					> (this->parameters.size);
			this->matrix_temperature_output_host[stream] = std::vector < full
					> (this->parameters.size);

			this->matrix_power_device[stream] = this->matrix_power_host[stream];
			this->matrix_temperature_input_device[stream] =
					this->matrix_temperature_input_host[stream];
			this->matrix_temperature_output_device[stream] =
					this->matrix_temperature_output_host[stream];

			rad::checkFrameworkErrors(
					cudaStreamCreateWithFlags(&this->streams[stream],
							cudaStreamNonBlocking));

			this->out_vector[stream] = std::vector < full
					> (this->parameters.size);

			this->matrix_temperature_output_incomplete_host[stream] =
					std::vector < incomplete > (this->parameters.size);
			this->matrix_temperature_output_incomplete_device[stream] =
					this->matrix_temperature_output_incomplete_host[stream];

		}

		this->gold_temperature = std::vector < full > (this->parameters.size);
	}

	void copy_from_gpu() {
		for (int stream = 0; stream < this->streams.size(); stream++) {
			rad::DeviceVector<full>* output[2] = {
					&this->matrix_temperature_input_device[stream],
					&this->matrix_temperature_output_device[stream] };

			this->out_vector[stream] =
					output[this->output_index[stream]]->to_vector();

			this->matrix_temperature_output_incomplete_host[stream] =
					this->matrix_temperature_output_incomplete_device[stream].to_vector();
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

	void sync() {
		for (auto stream : this->streams) {
			rad::checkFrameworkErrors(cudaStreamSynchronize(stream));
		}
		rad::checkFrameworkErrors(cudaDeviceSynchronize());
		rad::checkFrameworkErrors(cudaPeekAtLastError());
	}

	virtual ~DataManagement() {
		//Only destroy stream, the others will be automatically destroyed
		for (auto stream : this->streams) {
			rad::checkFrameworkErrors(cudaStreamDestroy(stream));
		}
	}

	// Returns true if no errors are found. False if otherwise.
	void check_output_errors() {
		if (this->parameters.generate == true) {
			return;
		}

		size_t& host_errors = this->log.error_count;
		size_t detected_errors = 0;
		double max_t = -22222;
		for (int stream = 0; stream < this->parameters.nstreams; stream++) {

#pragma omp parallel for shared(host_errors)
			for (int i = 0; i < this->parameters.grid_rows; i++) {
				for (int j = 0; j < this->parameters.grid_cols; j++) {
					int index = i * this->parameters.grid_rows + j;

					double valGold = this->gold_temperature[index];
					double valOutput = this->out_vector[stream][index];
					double valOutputIncomplete =
							this->matrix_temperature_output_incomplete_host[stream][index];
					bool isDiff = cmp(valOutput, valOutputIncomplete);
					double diff = std::fabs(valOutput - valOutputIncomplete);
					max_t = std::max(diff, max_t);

					detected_errors += size_t(isDiff);
					if (valGold != valOutput || !isDiff) {
#pragma omp critical
						{
							char error_detail[200];
							snprintf(error_detail, 200,
									"stream: %d, p: [%d, %d], r: %1.20e, e: %1.20e s: %1.20e",
									stream, i, j, valOutput, valGold,
									valOutputIncomplete);
							this->log.log_error(std::string(error_detail));
							if (this->parameters.verbose && (host_errors < 10))
								std::cout << error_detail << std::endl;

						}
					}
				}
			}
		}
//		std::ofstream of("test.txt", std::ofstream::out | std::ofstream::app);
//
//		 of << "BLOCK " << CHECKBLOCK << " MAX DIFF " << max_t << std::endl;
//		 of.close();

		if (detected_errors != 0) {
			std::string error_detail;
			error_detail = "detected_dmr_errors: "
					+ std::to_string(detected_errors);

			this->log.log_error(error_detail);
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
			exit (EXIT_FAILURE);
		}

		if (!power_file.is_open()) {
			std::cerr << "The power file was not opened" << std::endl;
			exit (EXIT_FAILURE);
		}

		if (this->parameters.generate == false) {
			if (gold_file.is_open() == false) {
				std::cerr << "The gold file was not opened" << std::endl;
				exit (EXIT_FAILURE);
			}
			// reading from gold
			gold_file.read((char*) this->gold_temperature.data(),
					sizeof(full) * this->parameters.size);
		}

		std::vector < full > temperature(this->parameters.size);
		std::vector < full > power(this->parameters.size);

		for (int i = 0; i < this->parameters.grid_rows; i++) {
			for (int j = 0; j < this->parameters.grid_cols; j++) {
				if (temp_file.eof()) {
					std::cerr << "[" << i << "," << j << "] size: "
							<< this->parameters.size << std::endl;
					std::cerr << "not enough lines in temp file" << std::endl;
					exit (EXIT_FAILURE);
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
					exit (EXIT_FAILURE);
				}

				power_file >> power_val;

				power[i * this->parameters.grid_cols + j] = full(power_val);

				if (power_val == 0)
					num_zeros++;
				if (std::isnan(power_val))
					num_nans++;
			}
		}
		if (this->parameters.verbose) {
			std::cout << "Zeros in the input: " << num_zeros << std::endl;
			std::cout << "NaNs in the input: " << num_nans << std::endl;
		}
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
			exit (EXIT_FAILURE);
		}

		gold_file.write((char*) this->out_vector[0].data(),
				sizeof(full) * this->parameters.size);

		gold_file.close();

		int nan = 0;
		int zero = 0;
		for (auto n : this->out_vector[0]) {
			if (std::isnan(float(n)))
				nan++;
			if (float(n) == 0)
				zero++;
		}

		if (this->parameters.verbose) {
			std::cout << "Gold Zeros in the output: " << zero << std::endl;
			std::cout << "Gold NaNs in the output: " << nan << std::endl;
		}
	}
private:
	bool cmp(const double lhs, const double rhs) {
		const double diff = std::fabs(lhs - rhs);
		const double zero = double(ZERO_FLOAT);
		if (diff > zero) {
			return false;
		}
		return true;
	}

};

#endif /* DATAMANAGEMENT_H_ */
