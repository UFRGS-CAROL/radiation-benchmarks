/*
 * register_file_kernels.cu
 *
 *  Created on: Feb 2, 2019
 *      Author: carol
 */

#include <cstring>
#include <ctime>
#include <bitset>

#include "RegisterFile.h"
#include "Parameters.h"
#include "utils.h"
#include "register_kernel.h"

RegisterFile::RegisterFile(const Parameters& parameters) :
		Memory<uint32>(parameters) {
	//Kepler and Volta have
	//256KB registers per SM
	//256 threads
	this->number_of_threads = parameters.registers_per_block / RF_SIZE;

	uint32 out_size = parameters.number_of_sms * parameters.registers_per_block;

	this->number_of_sms = parameters.number_of_sms;
	this->input_host_1 = std::vector<uint32>(RF_SIZE);
	this->output_host_1 = std::vector<uint32>(out_size);
}
void RegisterFile::test(const uint32& mem) {
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);

	rad::DeviceVector<uint32> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint32> output_device_1 = this->output_host_1;
	std::cout << output_device_1.size() << std::endl;
	double start = rad::mysecond();

	test_register_file_kernel<<<number_of_sms, number_of_threads>>>(
			output_device_1.data(), input_device_1.data(), cycles);

	cuda_check(cudaDeviceSynchronize());
	double end = rad::mysecond();

	this->output_host_1 = output_device_1.to_vector();
}

std::string RegisterFile::error_detail(uint32 i, uint32 e, uint32 r,
		uint64 hits, uint64 misses, uint64 false_hits) {
	std::string error_detail = "";
	error_detail += " i:" + std::to_string(i);
	error_detail += " register:R" + std::to_string(i % 256);
	error_detail += " e:" + std::to_string(e);
	error_detail += " r:" + std::to_string(r);
	return error_detail;
}

void RegisterFile::call_checker(const std::vector<uint32>& v1,
		const uint32& valGold, Log& log, uint64 hits, uint64 misses,
		uint64 false_hits, bool verbose) {
	this->check_output_errors(v1.data(), valGold, log, hits, misses, false_hits,
			verbose, v1.size());
}

