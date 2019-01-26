/*
 * NVMLWrapper.h
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#ifndef NVMLWRAPPER_H_
#define NVMLWRAPPER_H_

#include <nvml.h>

class NVMLWrapper {
private:
	nvmlReturn_t init_ok;
	nvmlReturn_t shutdown_ok;
	unsigned device_index;

public:
	NVMLWrapper(unsigned device_index);
	virtual ~NVMLWrapper();

	void start_collecting_data();

	void end_collecting_data();
};

#endif /* NVMLWRAPPER_H_ */
