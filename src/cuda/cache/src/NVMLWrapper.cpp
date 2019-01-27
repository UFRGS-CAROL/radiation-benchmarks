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
		result = nvmlDeviceGetComputeRunningProcesses(*device, info_count,
				&infos);
		for (auto t : info_count)
			if (t != 0)
				std::cout << "COUNT I " << t << " mem size "
						<< infos.usedGpuMemory << std::endl;

		//for future uses
		// nvmlReturn_t nvmlDeviceClearEccErrorCounts ( nvmlDevice_t device, nvmlEccCounterType_t counterType )
		//		nvmlMemoryErrorType_t
//		NVML_MEMORY_ERROR_TYPE_CORRECTED
//		NVML_MEMORY_ERROR_TYPE_UNCORRECTED
//		NVML_MEMORY_ERROR_TYPE_COUNT

//		nvmlEccCounterType_t counterType;
		//NVML_VOLATILE_ECC - reset on reboot
		//NVML_AGGREGATE_ECC - persistent
		//NVML_ECC_COUNTER_TYPE_COUNT

		for (auto error_type : { NVML_MEMORY_ERROR_TYPE_CORRECTED,
				NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_MEMORY_ERROR_TYPE_COUNT })
			for (auto counter_type : { NVML_VOLATILE_ECC, NVML_AGGREGATE_ECC,
					NVML_ECC_COUNTER_TYPE_COUNT }) {
				nvmlEccErrorCounts_t ecc_counts;
				result = nvmlDeviceGetDetailedEccErrors(*device, error_type,
						counter_type, &ecc_counts);
				std::cout << "ERROR TYPE " << error_type << " COUNTER_TYPE "
						<< counter_type << " MEM " << ecc_counts.deviceMemory
						<< " L1 " << ecc_counts.l1Cache << " L2 "
						<< ecc_counts.l2Cache << " RF "
						<< ecc_counts.registerFile << std::endl;
			}

		size_t last_seen_timestamp = get_time_since_epoch();
		for (auto sample_type : { NVML_TOTAL_POWER_SAMPLES,
				NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES,
				NVML_ENC_UTILIZATION_SAMPLES, NVML_DEC_UTILIZATION_SAMPLES,
				NVML_PROCESSOR_CLK_SAMPLES, NVML_MEMORY_CLK_SAMPLES,
				NVML_SAMPLINGTYPE_COUNT }) {
			nvmlValueType_t sample_val_type;
			unsigned sample_count;

			result = nvmlDeviceGetSamples(*device, sample_type,
					last_seen_timestamp, &sample_val_type, &sample_count,
					NULL);
			std::vector<nvmlSample_t> samples_array(sample_count);

			result = nvmlDeviceGetSamples(*device, sample_type,
					last_seen_timestamp, &sample_val_type, &sample_count,
					samples_array.data());
			std::cout << "SAMPLE TYPE " << sample_type << " SAMPLE VAL TYPE "
					<< sample_val_type << " sample count " << sample_count
					<< std::endl;

			for (auto st : samples_array) {
				if (st.sampleValue.dVal || st.sampleValue.sllVal
						|| st.sampleValue.uiVal || st.sampleValue.ulVal
						|| st.sampleValue.ullVal)
					std::cout << "samples: sample timestamp " << st.timeStamp
							<< " sample val " << st.sampleValue.dVal << " "
							<< st.sampleValue.uiVal << " "
							<< st.sampleValue.ulVal << " "
							<< st.sampleValue.ullVal << std::endl;
			}

		}
//		 nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus ( nvmlDevice_t device, nvmlEnableState_t* isPending )
//		 nvmlReturn_t nvmlDeviceGetTotalEccErrors ( nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts )
//		 nvmlReturn_t nvmlDeviceGetViolationStatus ( nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime )
//		 nvmlReturn_t nvmlDeviceRegisterEvents ( nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set )
//		 nvmlReturn_t nvmlDeviceGetSupportedEventTypes ( nvmlDevice_t device, unsigned long long* eventTypes )

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
