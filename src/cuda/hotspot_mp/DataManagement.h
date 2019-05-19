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

template<typename full>
struct DataManagement {
	std::vector<std::vector<full>> matrix_temperature_input_host;
	std::vector<std::vector<full>> matrix_temperature_output_host;
	std::vector<std::vector<full>> matrix_power_host;

	std::vector<full> gold_temperature;

	std::vector<cudaStream_t> streams;

	int size;

	// Alloc for multiple streams
	std::vector<DeviceVector<full> > matrix_temperature_input_device;
	std::vector<DeviceVector<full> > matrix_temperature_output_device;
	std::vector<DeviceVector<full> > matrix_power_device;

	DataManagement(int n_streams, int size) :
			size(size) {

		this->matrix_power_device.resize(n_streams);
		this->matrix_temperature_input_device.resize(n_streams);
		this->matrix_temperature_output_device.resize(n_streams);

		this->matrix_power_host.resize(n_streams);
		this->matrix_temperature_input_host.resize(n_streams);
		this->matrix_temperature_output_host.resize(n_streams);

		this->streams = std::vector<cudaStream_t>(n_streams);
		for (int stream = 0; stream < n_streams; stream++) {

			this->matrix_power_host[stream] = std::vector<full>(size);
			this->matrix_temperature_input_host[stream] = std::vector<full>(
					size);
			this->matrix_temperature_output_host[stream] = std::vector<full>(
					size);

			this->matrix_power_device[stream] = this->matrix_power_host[stream];
			this->matrix_temperature_input_device[stream] =
					this->matrix_temperature_input_host[stream];
			this->matrix_temperature_output_device[stream] =
					this->matrix_temperature_output_host[stream];

			checkFrameworkErrors(
					cudaStreamCreateWithFlags(&this->streams[stream],
							cudaStreamNonBlocking));

		}

		this->gold_temperature = std::vector<full>(size);

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
//		double timestamp = Log.mysecond();
//		// =================== Read all files
//		int i, j;
//		FILE *ftemp, *fpower, *fgold;
//		char str[STR_SIZE];
//		float val;
//		int num_zeros = 0;
//		int num_nans = 0;
//
//		if ((ftemp = fopen(this->tfile, "r")) == 0)
//			fatal(params, "The temp file was not opened");
//		if ((fpower = fopen(this->pfile, "r")) == 0)
//			fatal(params, "The power file was not opened");
//
//		if (!(this->generate))
//			if ((fgold = fopen(this->ofile, "rb")) == 0)
//				fatal(params, "The gold was not opened");
//
//		for (i = 0; i <= (this->grid_rows) - 1; i++) {
//			for (j = 0; j <= (this->grid_cols) - 1; j++) {
//				if (!fgets(str, STR_SIZE, ftemp)) {
//					fatal(params, "not enough lines in temp file");
//				}
//				if (feof(ftemp)) {
//					printf("[%d,%d] size: %d ", i, j, this->grid_rows);
//					fatal(params, "not enough lines in temp file");
//				}
//				if ((sscanf(str, "%f", &val) != 1))
//					fatal(params, "invalid temp file format");
//
//				this->temperature_input[i * (this->grid_cols) + j] =
//						tested_type_host(val);
//
//				if (tested_type_host(val) == 0)
//					num_zeros++;
//				if (isnan(tested_type_host(val)))
//					num_nans++;
//
//				if (!fgets(str, STR_SIZE, fpower)) {
//					fatal(params, "not enough lines in power file");
//				}
//				if (feof(fpower))
//					fatal(params, "not enough lines in power file");
//				if ((sscanf(str, "%f", &val) != 1))
//					fatal(params, "invalid power file format");
//
//				this->power[i * (this->grid_cols) + j] = tested_type_host(val);
//
//				if (tested_type_host(val) == 0)
//					num_zeros++;
//				if (isnan(tested_type_host(val)))
//					num_nans++;
//
//				if (!(this->generate)) {
//					assert(
//							fread(
//									&(this->gold_temperature[i
//											* (this->grid_cols) + j]),
//									sizeof(tested_type), 1, fgold) == 1);
//				}
//			}
//		}
//
//		printf("Zeros in the input: %d\n", num_zeros);
//		printf("NaNs in the input: %d\n", num_nans);
//
//		// =================== FAULT INJECTION
//		if (this->fault_injection) {
//			this->in_temperature[32] = 6.231235;
//			printf("!!!!!!!!! Injected error: in_temperature[32] = %f\n",
//					(double) this->temperature_input[32]);
//		}
//		// ==================================
//
//		fclose(ftemp);
//		fclose(fpower);
//		if (!(this->generate))
//			fclose(fgold);
//
//		if (this->verbose)
//			std::cout << "readInput time: " << Log.mysecond() - timestamp
//					<< std::endl;
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
