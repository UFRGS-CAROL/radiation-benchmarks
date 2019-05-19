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
#include <cmath>

template<typename full>
struct DataManagement {
	std::vector<std::vector<full>> matrix_temperature_input_host;
	std::vector<std::vector<full>> matrix_temperature_output_host;
	std::vector<std::vector<full>> matrix_power_host;

	std::vector<full> gold_temperature;
	std::vector<cudaStream_t> streams;
	const Parameters& parameters;

	// Alloc for multiple streams
	std::vector<DeviceVector<full> > matrix_temperature_input_device;
	std::vector<DeviceVector<full> > matrix_temperature_output_device;
	std::vector<DeviceVector<full> > matrix_power_device;

	DataManagement(Parameters& parameters) :
			parameters(parameters) {

		this->matrix_power_device.resize(this->parameters.nstreams);
		this->matrix_temperature_input_device.resize(this->parameters.nstreams);
		this->matrix_temperature_output_device.resize(
				this->parameters.nstreams);

		this->matrix_power_host.resize(this->parameters.nstreams);
		this->matrix_temperature_input_host.resize(this->parameters.nstreams);
		this->matrix_temperature_output_host.resize(this->parameters.nstreams);

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

		}

		this->gold_temperature = std::vector<full>(this->parameters.size);

	}

	void copy_from_gpu() {
		for (int stream = 0; stream < this->streams.size(); stream++) {
			this->matrix_power_host[stream] =
					this->matrix_power_device[stream].to_vector();
			this->matrix_temperature_input_host[stream] =
					this->matrix_temperature_input_device[stream].to_vector();
			this->matrix_temperature_output_host[stream] =
					this->matrix_temperature_output_device[stream].to_vector();
		}
	}

	void reload() {
		for (int stream = 0; stream < this->streams.size(); stream++) {
			this->matrix_power_device[stream] = this->matrix_power_host[stream];
			this->matrix_temperature_input_device[stream] =
					this->matrix_temperature_input_host[stream];
			this->matrix_temperature_output_device[stream].fill(0);
		}
	}

	virtual ~DataManagement() {
		//Only destroy stream, the others will be automatically destroyed
		for (auto stream : this->streams) {
			checkFrameworkErrors(cudaStreamDestroy(stream));
		}
	}

	// Returns true if no errors are found. False if otherwise.
	int check_output_errors() {
		//	int host_errors = 0;
		//
		//#pragma omp parallel for shared(host_errors)
		//	for (int i = 0; i < setup_parameters->grid_rows; i++) {
		//		for (int j = 0; j < setup_parameters->grid_cols; j++) {
		//			int index = i * setup_parameters->grid_rows + j;
		//
		//			register tested_type_host valGold =
		//					setup_parameters->gold_temperature[index];
		//			register tested_type_host valOutput =
		//					setup_parameters->out_temperature[index];
		//
		//			if (valGold != valOutput) {
		//#pragma omp critical
		//				{
		//					char error_detail[150];
		//					snprintf(error_detail, 150,
		//							"stream: %d, p: [%d, %d], r: %1.20e, e: %1.20e",
		//							streamIdx, i, j, (double) valOutput,
		//							(double) valGold);
		//					if (setup_parameters->verbose && (host_errors < 10))
		//						printf("%s\n", error_detail);
		//#ifdef LOGS
		//					if (!setup_parameters->generate)
		//					log_error_detail(error_detail);
		//#endif
		//					host_errors++;
		//
		//				}
		//			}
		//		}
		//	}
		//
		//#ifdef LOGS
		//	if (!setup_parameters->generate) {
		//		log_error_count(host_errors);
		//	}
		//#endif
		//	if ((host_errors != 0) && (!setup_parameters->verbose))
		//		printf("#");
		//	if ((host_errors != 0) && (setup_parameters->verbose))
		//		printf("Output errors: %d\n", host_errors);
		//
		//	return (host_errors == 0);
		return 0;

	}

	void readInput() {
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
			std::cerr << ("The temp file was not opened");
			exit(EXIT_FAILURE);
		}

		if (!power_file.is_open()) {
			std::cerr << ("The power file was not opened");
			exit(EXIT_FAILURE);
		}

		if (this->parameters.generate) {
			if (!gold_file.is_open()) {
				std::cerr << ("The gold file was not opened");
				exit(EXIT_FAILURE);
			}

			gold_file.read((char*)this->gold_temperature.data(),
					sizeof(full) * this->parameters.size);
		}

		std::vector<full> temperature(this->parameters.size);
		std::vector<full> power(this->parameters.size);

		// reading from gold

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

	void writeOutput() {
//		// =================== Write output to gold file
//		int i, j;
//		FILE *fgold;
//		// char str[STR_SIZE];
//		int num_zeros = 0;
//		int num_nans = 0;
//
//		if ((fgold = fopen(this->ofile, "wb")) == 0)
//			fatal(params, "The gold was not opened");
//
//		for (i = 0; i <= (this->grid_rows) - 1; i++) {
//			for (j = 0; j <= (this->grid_cols) - 1; j++) {
//				// =======================
//				//HARDENING AGAINST BAD BOARDS
//				//-----------------------------------------------------------------------------------
//
//				if (this->temperature_output[i * (this->grid_cols) + j] == 0)
//					num_zeros++;
//
//				if (isnan(this->out_temperature[i * (this->grid_cols) + j]))
//					num_nans++;
//
//				//-----------------------------------------------------------------------------------
//				fwrite(&(this->temperature_output[i * (this->grid_cols) + j]),
//						sizeof(tested_type), 1, fgold);
//			}
//		}
//		fclose(fgold);
//		printf("Zeros in the output: %d\n", num_zeros);
//		printf("NaNs in the output: %d\n", num_nans);
	}
};

#endif /* DATAMANAGEMENT_H_ */
