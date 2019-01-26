/*
 * NVMLWrapper.h
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#ifndef NVMLWRAPPER_H_
#define NVMLWRAPPER_H_

#include <nvml.h>
#include <string>

class NVMLWrapper {
private:

	unsigned device_index;
	std::string device_name;
	std::string driver_version;
	std::string nvml_version;
	nvmlDevice_t device;

	void check_nvml_return(std::string info, nvmlReturn_t result);
public:
	NVMLWrapper(unsigned device_index);
	virtual ~NVMLWrapper();

	void start_collecting_data();

	void end_collecting_data();
};

#endif /* NVMLWRAPPER_H_ */
