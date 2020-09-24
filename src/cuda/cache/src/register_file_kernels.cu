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
#include "register_kernel_volta.h"

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

void RegisterFile::test(const uint64& mem_) {
	uint32 mem = mem_;
	//Set values to GPU
	std::fill(this->input_host_1.begin(), this->input_host_1.end(), mem);
	rad::DeviceVector<uint32> input_device_1 = this->input_host_1;
	rad::DeviceVector<uint32> output_device_1 = this->output_host_1;
	rad::DeviceVector<uint32> output_device_2, output_device_3;
	switch (this->device) {
	case K20:
	case K40:
		test_register_file_kernel<<<number_of_sms, number_of_threads>>>(
				output_device_1.data(), input_device_1.data(), cycles);
		break;
	case TITANV:
	case XAVIER:
		output_device_2 = this->output_host_1;
		output_device_3 = this->output_host_1;
		uint32 zero_or_one = (mem_ == 0);
		test_register_file_kernel_volta<<<number_of_sms, number_of_threads>>>(
				output_device_1.data(), output_device_2.data(),
				output_device_3.data(), zero_or_one, this->cycles);
		break;
	};

    rad::checkFrameworkErrors(cudaPeekAtLastError());
    rad::checkFrameworkErrors(cudaDeviceSynchronize());
	this->output_host_1 = output_device_1.to_vector();
	this->output_host_2 = output_device_2.to_vector();
	this->output_host_3 = output_device_3.to_vector();

}

std::string RegisterFile::error_detail(uint64 i, uint64 e, uint64 r, int64 hits,
		int64 misses, int64 false_hits) {
	std::string error_detail = "";
	error_detail += " i:" + std::to_string(i);
	error_detail += " register:R" + std::to_string(i % 256);
	error_detail += " e:" + std::to_string(e);
	error_detail += " r:" + std::to_string(r);
	return error_detail;
}

bool RegisterFile::call_checker(uint64& gold, rad::Log& log, int64& hits,
		int64& misses, int64& false_hits, bool verbose) {
	uint32* out_ptr1 = (uint32*) this->output_host_1.data();
	uint32* out_ptr2 = (uint32*) this->output_host_2.data();
	uint32* out_ptr3 = (uint32*) this->output_host_3.data();

	uint32 gold_ = gold;

	return this->check_output_errors(out_ptr1, out_ptr2, out_ptr3, gold_, log,
			hits, misses, false_hits, this->output_host_1.size(), verbose);
}
