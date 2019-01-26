/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include "NVMLWrapper.h"
#include "utils.h"

NVMLWrapper::NVMLWrapper() {
	this->init_ok = nvmlInit();
	if(this->init_ok != NVML_SUCCESS){
		error("Cannot initialize NVML library");
	}
}

NVMLWrapper::~NVMLWrapper() {
	this->shutdown_ok = nvmlShutdown();
}

