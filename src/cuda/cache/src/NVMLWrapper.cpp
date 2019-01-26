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
	char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
	result = nvmlDeviceGetHandleByIndex(this->device_index, &this->device);
	this->check_nvml_return("get handle", result);
	result = nvmlDeviceGetName(this->device, device_name,
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

	this->device_name = device_name;
	this->driver_version = driver_version;
	this->nvml_version = nvml_version;
}

NVMLWrapper::~NVMLWrapper() {
	nvmlReturn_t result = nvmlShutdown();
	this->check_nvml_return("initialize NVML library", result);
}

void NVMLWrapper::start(nvmlDevice_t* device) {
	for (int i = 0; i < 10; i++) {
		sleep(1);
		unsigned clock;
		nvmlReturn_t result = nvmlDeviceGetClockInfo(*device,
				NVML_CLOCK_GRAPHICS, &clock);

		std::cout << "GRAPHICS " << clock << std::endl;
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_MEM, &clock);

		std::cout << "MEMORY " << clock << std::endl;
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_SM, &clock);

		std::cout << "SM " << clock << std::endl;
		result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_COUNT, &clock);
		std::cout << "COUNT " << clock << std::endl;

		//get compute mode
		nvmlComputeMode_t mode;
		result = nvmlDeviceGetComputeMode(*device, &mode);
		std::cout << "COMPUTE MODE " << mode << std::endl;

		//get info counts
		unsigned info_count[10];
		nvmlProcessInfo_t infos;
		result = nvmlDeviceGetComputeRunningProcesses (*device, info_count, &infos);
		for(auto t : info_count)
			std::cout << "COUNT I " << t << std::endl;

	}
}

void NVMLWrapper::start_collecting_data() {
	this->profiler = std::thread(NVMLWrapper::start, &this->device);
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
	this->profiler.join();
}

void NVMLWrapper::print_device_info() {
	std::cout << "Device name: " << this->device_name << std::endl
			<< "Driver version: " << this->driver_version << std::endl
			<< "NVML version: " << this->nvml_version << std::endl;
}
