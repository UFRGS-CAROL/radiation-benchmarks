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
public:
	NVMLWrapper();
	virtual ~NVMLWrapper();
};

#endif /* NVMLWRAPPER_H_ */
