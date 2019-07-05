/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include <condition_variable>

#include <algorithm>
#include <iostream>
#include <vector>

#include "NVMLWrapper.h"

namespace rad {

void check_nvml_return(std::string info, nvmlReturn_t result, unsigned device =
		0) {
	if (NVML_SUCCESS != result) {
		/*error(
		 "Failed to " + info + " from device " + std::to_string(device)
		 + " error " + nvmlErrorString(result));*/
		std::cerr
				<< "Failed to " + info + " from device "
						+ std::to_string(device) + " error "
						+ nvmlErrorString(result);
	}
}

NVMLWrapper::NVMLWrapper(unsigned device_index, std::string& output_file) :
		Profiler(device_index, output_file), _nvml_set(nullptr), _nvml_device(
				nullptr) {
	this->_thread_profiler = std::thread(NVMLWrapper::data_colector,
			&this->_nvml_device, &this->data_for_iteration, &this->_mutex_lock,
			&this->_is_locked, &this->_thread_running);
}

NVMLWrapper::~NVMLWrapper() {
	_thread_running = false;
	this->_thread_profiler.join();
}

void NVMLWrapper::data_colector(nvmlDevice_t* device,
		std::deque<std::string>* it_data, std::mutex* mutex_lock,
		std::atomic<bool>* is_locked, std::atomic<bool>* _thread_running) {
	nvmlReturn_t result;

	while (*_thread_running) {
		mutex_lock->lock();

		if (*is_locked == false) {
			std::string output = "";
			//-----------------------------------------------------------------------
			/**
			 * Device and application clocks
			 * May be useful in the future

			 for (auto clock_type : { NVML_CLOCK_GRAPHICS, NVML_CLOCK_MEM,
			 NVML_CLOCK_SM }) {
			 //Get DEVICE Clocks
			 unsigned dev_clock, app_clock;
			 result = nvmlDeviceGetClockInfo(*device, clock_type,
			 &dev_clock);

			 //Get Application clocks
			 result = nvmlDeviceGetApplicationsClock(*device, clock_type,
			 &app_clock);

			 output += std::to_string(dev_clock) + ","
			 + std::to_string(app_clock) + ",";
			 }
			 */
			//-----------------------------------------------------------------------
			//Get ECC errors
			nvmlComputeMode_t compute_mode;
			result = nvmlDeviceGetComputeMode(*device, &compute_mode);
			std::cout << result << std::endl;
			for (auto error_type : { NVML_MEMORY_ERROR_TYPE_CORRECTED,
					NVML_MEMORY_ERROR_TYPE_UNCORRECTED }) {
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
					output += std::to_string(total_eec_errors) + ",";
				}
			}

			//-----------------------------------------------------------------------
			//Get Performance state P0 to P12
			nvmlPstates_t p_state;
			result = nvmlDeviceGetPerformanceState(*device, &p_state);
			output += std::to_string(p_state) + ",";

			//-----------------------------------------------------------------------
			//Clocks throttle
			unsigned long long clocks_throttle_reasons;
			result = nvmlDeviceGetCurrentClocksThrottleReasons(*device,
					&clocks_throttle_reasons);
			output += std::to_string(clocks_throttle_reasons) + ",";

			//-----------------------------------------------------------------------
			//Get utilization on GPU
			nvmlUtilization_t utilization;
			result = nvmlDeviceGetUtilizationRates(*device, &utilization);
			output += std::to_string(utilization.gpu) + ","
					+ std::to_string(utilization.memory) + ",";

			//-----------------------------------------------------------------------
			//Get retired pages
			for (auto cause : {
					NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
					NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS }) {
				unsigned int page_count = 0;
				unsigned long long *addresses = nullptr;
				device = 0;
				nvmlDeviceGetRetiredPages(*device, cause, &page_count,
						addresses);
				output += std::to_string(page_count) + ",";
			}

			//-----------------------------------------------------------------------
			//Get retired pages pending status
			nvmlEnableState_t is_pending;
			result = nvmlDeviceGetRetiredPagesPendingStatus(*device,
					&is_pending);
			output += std::to_string(is_pending) + ",";

			//-----------------------------------------------------------------------
			//Get GPU temperature
			unsigned int temperature;
			result = nvmlDeviceGetTemperature(*device, NVML_TEMPERATURE_GPU,
					&temperature);
			output += std::to_string(temperature) + ",";

			//-----------------------------------------------------------------------
			//Get GPU power
			unsigned int power;
			result = nvmlDeviceGetPowerUsage(*device, &power);
			output += std::to_string(power);

//			auto timestamp = std::chrono::system_clock::to_time_t(
//					std::chrono::system_clock::now());
//			output += std::to_string(timestamp);

			it_data->push_back(output);
		}
		mutex_lock->unlock();
		std::this_thread::sleep_for(std::chrono::microseconds(PROFILER_SLEEP));
	}
}

void NVMLWrapper::start_profile() {
	nvmlReturn_t result = nvmlInit();
	check_nvml_return("initialize NVML library", result);
	result = nvmlDeviceGetHandleByIndex(this->_device_index,
			&this->_nvml_device);
	result = nvmlEventSetCreate(&this->_nvml_set);
	result = nvmlDeviceRegisterEvents(this->_nvml_device, nvmlEventTypeAll,
			this->_nvml_set);

	this->_mutex_lock.lock();
	this->data_for_iteration.clear();
	this->_mutex_lock.unlock();

	this->_is_locked = false;
}

void NVMLWrapper::end_profile() {
	this->_mutex_lock.lock();
	this->_is_locked = true;
	this->_mutex_lock.unlock();

	nvmlReturn_t result;
	result = nvmlEventSetFree(this->_nvml_set);
	result = nvmlShutdown();
	check_nvml_return("shutdown NVML library", result);
}

std::deque<std::string> NVMLWrapper::get_data_from_iteration() {
	auto last = std::unique(this->data_for_iteration.begin(),
			this->data_for_iteration.end());
	this->data_for_iteration.erase(last, this->data_for_iteration.end());
	return this->data_for_iteration;
}

}

