/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include "NVMLWrapper.h"
#include "utils.h"

#include <iostream>

const char * convertToComputeModeString(nvmlComputeMode_t mode) {
	switch (mode) {
	case NVML_COMPUTEMODE_DEFAULT:
		return "Default";
	case NVML_COMPUTEMODE_EXCLUSIVE_THREAD:
		return "Exclusive_Thread";
	case NVML_COMPUTEMODE_PROHIBITED:
		return "Prohibited";
	case NVML_COMPUTEMODE_EXCLUSIVE_PROCESS:
		return "Exclusive Process";
	default:
		return "Unknown";
	}
}

NVMLWrapper::NVMLWrapper(unsigned device_index) :
		device_index(device_index) {
	this->init_ok = nvmlInit();
	if (this->init_ok != NVML_SUCCESS) {
		error(
				"Cannot initialize NVML library, error: "
						+ std::to_string(this->init_ok));
	}

	this->shutdown_ok = NVML_ERROR_DRIVER_NOT_LOADED;

}

NVMLWrapper::~NVMLWrapper() {
	this->shutdown_ok = nvmlShutdown();
	if (this->shutdown_ok != NVML_SUCCESS) {
		error(
				"Cannot initialize NVML library, error: "
						+ std::to_string(this->shutdown_ok));
	}
}

void NVMLWrapper::start_collecting_data() {
	nvmlReturn_t result;
	nvmlDevice_t device;
	char name[NVML_DEVICE_NAME_BUFFER_SIZE];
	nvmlPciInfo_t pci;
	nvmlComputeMode_t compute_mode;

	// Query for device handle to perform operations on a device
	// You can also query device handle by other features like:
	// nvmlDeviceGetHandleBySerial
	// nvmlDeviceGetHandleByPciBusId
	result = nvmlDeviceGetHandleByIndex(this->device_index, &device);
	if (NVML_SUCCESS != result) {
		error(
				"Failed to get handle for device "
						+ std::to_string(this->device_index) + " "
						+ std::to_string(result));
	}

	result = nvmlDeviceGetName(device, name,
	NVML_DEVICE_NAME_BUFFER_SIZE);
	if (NVML_SUCCESS != result) {
		error(
				"Failed to get name of device "
						+ std::to_string(this->device_index) + " "
						+ std::to_string(result));
	}

	// pci.busId is very useful to know which device physically you're talking to
	// Using PCI identifier you can also match nvmlDevice handle to CUDA device.
	result = nvmlDeviceGetPciInfo(device, &pci);
	if (NVML_SUCCESS != result) {
		error(
				"Failed to get pci info for device "
						+ std::to_string(this->device_index) + " "
						+ std::to_string(result));
	}

	std::cout << this->device_index << " " << name << " " << pci.busId
			<< std::endl;

	// This is a simple example on how you can modify GPU's state
	result = nvmlDeviceGetComputeMode(device, &compute_mode);
	if (NVML_ERROR_NOT_SUPPORTED == result)
		std::cout << "\t This is not CUDA capable device\n";
	else if (NVML_SUCCESS != result) {
		error(
				"Failed to get compute mode for device "
						+ std::to_string(this->device_index) + " "
						+ std::to_string(result));
	} else {
		// try to change compute mode
		std::cout << "\t Changing device's compute mode from "
				<< convertToComputeModeString(compute_mode) << " to "
				<< convertToComputeModeString(NVML_COMPUTEMODE_PROHIBITED)
				<< std::endl;

		result = nvmlDeviceSetComputeMode(device, NVML_COMPUTEMODE_PROHIBITED);
		if (NVML_ERROR_NO_PERMISSION == result)
			error(
					"\t\t Need root privileges to do that: "
							+ std::to_string(result));
		else if (NVML_ERROR_NOT_SUPPORTED == result)
			std::cout
					<< "\t\t Compute mode prohibited not supported. You might be running on\n"
							"\t\t windows in WDDM driver model or on non-CUDA capable GPU.\n";
		else if (NVML_SUCCESS != result) {
			std::cout << "\t\t Failed to set compute mode for device "
					<< std::to_string(this->device_index) << " " << std::to_string(result);
		} else {
			std::cout << "\t Restoring device's compute mode back to "
					<< convertToComputeModeString(compute_mode) << std::endl;
			result = nvmlDeviceSetComputeMode(device, compute_mode);
			if (NVML_SUCCESS != result) {
				error(
						"\t\t Failed to restore compute mode for device "
								+ std::to_string(this->device_index) + " "
								+ std::to_string(result));
			}
		}
	}

}

void NVMLWrapper::end_collecting_data() {
}
