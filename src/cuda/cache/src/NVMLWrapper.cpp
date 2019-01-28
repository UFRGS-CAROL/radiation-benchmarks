/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include "NVMLWrapper.h"
#include "utils.h"

#include <iostream>
#include <vector>

void check_nvml_return(std::string info, nvmlReturn_t result, unsigned device =
		0) {
	if (NVML_SUCCESS != result) {
		error(
				"Failed to " + info + " from device " + std::to_string(device)
						+ " error " + nvmlErrorString(result));
	}
}

NVMLWrapper::NVMLWrapper(unsigned device_index) :
		device_index(device_index) {
	nvmlReturn_t result = nvmlInit();
	check_nvml_return("initialize NVML library", result);

	//getting device name
	char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
	result = nvmlDeviceGetHandleByIndex(this->device_index, &this->device);
	check_nvml_return("get handle", result);
	result = nvmlDeviceGetName(this->device, device_name,
	NVML_DEVICE_NAME_BUFFER_SIZE);

	//getting driver version
	char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
	result = nvmlSystemGetDriverVersion(driver_version,
	NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
	check_nvml_return("get driver version", result);

	// nvml version
	char nvml_version[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
	result = nvmlSystemGetNVMLVersion(nvml_version,
	NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);
	check_nvml_return("get nvml version", result);

	this->device_name = device_name;
	this->driver_version = driver_version;
	this->nvml_version = nvml_version;
}

NVMLWrapper::~NVMLWrapper() {
	nvmlReturn_t result = nvmlShutdown();
	check_nvml_return("initialize NVML library", result);
}

void NVMLWrapper::start(nvmlDevice_t* device) {
	nvmlEventSet_t set;
	nvmlReturn_t result = nvmlEventSetCreate(&set);
	result = nvmlDeviceRegisterEvents(*device, nvmlEventTypeAll, set);

	for (int i = 0; i < 10; i++) {
		sleep(1);
		unsigned graph_clock, mem_clock, sm_clock;
		std::string output = "";
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_GRAPHICS,
				&graph_clock);
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_MEM, &mem_clock);
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_SM, &sm_clock);
		output += std::to_string(graph_clock) + "," + std::to_string(mem_clock)
				+ "," + std::to_string(sm_clock) + ",";

		//get compute mode
		nvmlComputeMode_t compute_mode;
		result = nvmlDeviceGetComputeMode(*device, &compute_mode);

		for (auto error_type : { NVML_MEMORY_ERROR_TYPE_CORRECTED,
				NVML_MEMORY_ERROR_TYPE_UNCORRECTED })
			for (auto counter_type : { NVML_VOLATILE_ECC, NVML_AGGREGATE_ECC }) {
				nvmlEccErrorCounts_t ecc_counts;
				result = nvmlDeviceGetDetailedEccErrors(*device, error_type,
						counter_type, &ecc_counts);
				output += std::to_string(ecc_counts.deviceMemory) + ","
						+ std::to_string(ecc_counts.l1Cache) + ","
						+ std::to_string(ecc_counts.l2Cache) + ","
						+ std::to_string(ecc_counts.registerFile) + ",";
			}

		nvmlEnableState_t is_pending;
		result = nvmlDeviceGetRetiredPagesPendingStatus(*device, &is_pending);
		output += std::to_string(is_pending) + ",";

		unsigned long long ecc_counts;
		for (auto error_type : { NVML_MEMORY_ERROR_TYPE_CORRECTED,
				NVML_MEMORY_ERROR_TYPE_UNCORRECTED }) {
			for (auto counter_type : { NVML_VOLATILE_ECC, NVML_AGGREGATE_ECC }) {
				result = nvmlDeviceGetTotalEccErrors(*device, error_type,
						counter_type, &ecc_counts);
				output += std::to_string(ecc_counts);

			}
		}

		std::cout << "OUT STRING: " << output << std::endl;

	}
	result = nvmlEventSetFree(set);

}

void NVMLWrapper::start_collecting_data() {
	this->profiler = std::thread(NVMLWrapper::start, &this->device);
}

void NVMLWrapper::end_collecting_data() {
	this->profiler.join();
}

void NVMLWrapper::print_device_info() {
	std::cout << "Device name: " << this->device_name << std::endl
			<< "Driver version: " << this->driver_version << std::endl
			<< "NVML version: " << this->nvml_version << std::endl;
}
