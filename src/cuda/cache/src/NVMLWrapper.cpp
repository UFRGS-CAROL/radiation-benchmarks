/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include "NVMLWrapper.h"
#include "utils.h"

#include <iostream>

NVMLWrapper::NVMLWrapper(unsigned device_index) :
		device_index(device_index) {
	nvmlReturn_t result = nvmlInit();
	this->check_nvml_return("initialize NVML library", result);

	//getting device name
	std::string device_name;
	device_name.reserve(NVML_DEVICE_NAME_BUFFER_SIZE);
//	char            name[NVML_DEVICE_NAME_BUFFER_SIZE];
	result = nvmlDeviceGetHandleByIndex(this->device_index, &this->device);
	this->check_nvml_return("get handle", result);

	result = nvmlDeviceGetName(this->device,
			const_cast<char*>(this->device_name.c_str()),
			NVML_DEVICE_NAME_BUFFER_SIZE);

	//getting driver version
	char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
	result = nvmlSystemGetDriverVersion(driver_version,
	NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
	this->check_nvml_return("get driver version", result);

	// nvml version
	char nvml_version[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
	result = nvmlSystemGetNVMLVersion(nvml_version,
	NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);

	this->check_nvml_return("get nvml version", result);

}

NVMLWrapper::~NVMLWrapper() {
	nvmlReturn_t result = nvmlShutdown();
	this->check_nvml_return("initialize NVML library", result);
}

void NVMLWrapper::start_collecting_data() {

}

void NVMLWrapper::check_nvml_return(std::string info, nvmlReturn_t result) {
	if (NVML_SUCCESS != result) {
		error(
				"Failed to " + info + " from device "
						+ std::to_string(this->device_index) + " error "
						+ nvmlErrorString(result));
	}
}

void NVMLWrapper::end_collecting_data() {
}
