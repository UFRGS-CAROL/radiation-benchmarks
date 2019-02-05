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

static std::mutex mutex_lock;
static std::atomic<bool> is_locked;
static bool thread_running = true;

void check_nvml_return(std::string info, nvmlReturn_t result, unsigned device =
		0) {
	if (NVML_SUCCESS != result) {
		error(
				"Failed to " + info + " from device " + std::to_string(device)
						+ " error " + nvmlErrorString(result));
	}
}

NVMLWrapper::NVMLWrapper(unsigned device_index) :
		device_index(device_index), device(nullptr), set(nullptr) {
//	//getting device name
//	char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
//	check_nvml_return("get handle", result);
//	result = nvmlDeviceGetName(this->device, device_name,
//	NVML_DEVICE_NAME_BUFFER_SIZE);
//
//	//getting driver version
//	char driver_version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
//	result = nvmlSystemGetDriverVersion(driver_version,
//	NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE);
//	check_nvml_return("get driver version", result);
//
//	// nvml version
//	char nvml_version[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE];
//	result = nvmlSystemGetNVMLVersion(nvml_version,
//	NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE);
//	check_nvml_return("get nvml version", result);
//
//	this->device_name = device_name;
//	this->driver_version = driver_version;
//	this->nvml_version = nvml_version;
	this->profiler = std::thread(NVMLWrapper::data_colector, &this->device);
	is_locked = true;
}

NVMLWrapper::~NVMLWrapper() {
	thread_running = false;
	this->profiler.join();
}

std::string exec(const std::string& cmd) {
	std::array<char, 128> buffer;
	std::string result;
	std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
			pclose);
	if (!pipe) {
		throw std::runtime_error("popen() failed!");
	}
	while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
		result += buffer.data();
	}
	return result;
}

void get_data_from_nvidia_smi() {

}

void NVMLWrapper::data_colector(nvmlDevice_t* device) {
	nvmlReturn_t result;

	while (thread_running) {
		mutex_lock.lock();

		if (is_locked == false) {
			unsigned graph_clock, mem_clock, sm_clock;
			std::string output = "";
//			result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_GRAPHICS,
//					&graph_clock);
//			result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_MEM,
//					&mem_clock);
//			result = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_SM, &sm_clock);
//			output += std::to_string(graph_clock) + ","
//					+ std::to_string(mem_clock) + "," + std::to_string(sm_clock)
//					+ ",";

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

					unsigned long long total_eec_errors = 0;

					result = nvmlDeviceGetTotalEccErrors(*device, error_type,
							counter_type, &total_eec_errors);
					output += std::to_string(ecc_counts);
				}

			nvmlEnableState_t is_pending;
			result = nvmlDeviceGetRetiredPagesPendingStatus(*device,
					&is_pending);
			output += std::to_string(is_pending) + ",";

//			std::cout << "OUT STRING: " << output << std::endl;
		}
		mutex_lock.unlock();
		std::this_thread::sleep_for(std::chrono::microseconds(200));
	}
}

void NVMLWrapper::start_collecting_data() {
	nvmlReturn_t result = nvmlInit();
	check_nvml_return("initialize NVML library", result);
	result = nvmlDeviceGetHandleByIndex(this->device_index, &this->device);
	result = nvmlEventSetCreate(&this->set);
	result = nvmlDeviceRegisterEvents(this->device, nvmlEventTypeAll,
			this->set);

	is_locked = false;
}

void NVMLWrapper::end_collecting_data() {
	mutex_lock.lock();
	is_locked = true;
	mutex_lock.unlock();

	nvmlReturn_t result;
	result = nvmlEventSetFree(this->set);
	result = nvmlShutdown();
	check_nvml_return("shutdown NVML library", result);
}

