/*
 * NVMLWrapper.cpp
 *
 *  Created on: 25/01/2019
 *      Author: fernando
 */

#include <mutex>          // std::mutex
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <vector>

#include "JTX2Inst.h"
#include "jtx2/include/jtx1inst.h"

namespace rad {

static std::mutex mutex_lock;
static std::atomic<bool> is_locked;
static bool thread_running = true;

#define SLEEP_JTX2INST 500

JTX2Inst::JTX2Inst(unsigned device_index) :
		device_index(device_index) {
	this->profiler = std::thread(JTX2Inst::data_colector,
			&this->data_for_iteration);
	is_locked = true;
}

JTX2Inst::~JTX2Inst() {
	thread_running = false;
	this->profiler.join();
}

void JTX2Inst::data_colector(std::deque<std::string>* it_data) {
	unsigned int val;
	unsigned long rate;
	float convFromMilli;
	char *wunit, *aunit, *vunit;
	int convert;

	convert = 0;

	if (convert) {
		convFromMilli = 0.001;
		wunit = "W";
		aunit = "A";
		vunit = "V";
	} else {
		convFromMilli = 1;
		wunit = "mW";
		aunit = "mA";
		vunit = "mV";
	}
	while (thread_running) {
		mutex_lock.lock();

		if (is_locked == false) {
			std::string output = "";

			//***********************************************************************************************
			jtx1_get_ina3221(VDD_IN, POWER, &val);
			printf("[POWER] module power input: %.3f%s\n", convFromMilli * val,
					wunit);

			jtx1_get_ina3221(VDD_SYS_SOC, POWER, &val);
			printf("[POWER] SoC power: %.3f%s\n", convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_SYS_GPU, POWER, &val);
			printf("[POWER] GPU power rail: %.3f%s\n", convFromMilli * val,
					wunit);

			jtx1_get_ina3221(VDD_SYS_CPU, POWER, &val);
			printf("[POWER] CPU power rail: %.3f%s\n", convFromMilli * val,
					wunit);

			jtx1_get_ina3221(VDD_SYS_DDR, POWER, &val);
			printf("[POWER] (DDR) memory power rail: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_MUX, POWER, &val);
			printf("[POWER] (MUX) main carrier board power input: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_5V0_IO_SYS, POWER, &val);
			printf("[POWER] main carrier board 5V supply: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_3V3_SYS, POWER, &val);
			printf("[POWER] main carrier board 3.3V supply: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_3V3_IO_SLP, POWER, &val);
			printf("[POWER] carrier board 3.3V Sleep supply: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_1V8_IO, POWER, &val);
			printf("[POWER] main carrier board 1.8V supply: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_3V3_SYS_M2, POWER, &val);
			printf("[POWER] 3.3V supply for M.2 Key E connector: %.3f%s\n",
					convFromMilli * val, wunit);

			jtx1_get_ina3221(VDD_4V0_WIFI, POWER, &val);
			printf("[POWER] (WIFI) Antenna (?): %.3f%s\n", convFromMilli * val,
					wunit);
			printf("\n");
			//************************************************************************************************
			jtx1_get_ina3221(VDD_IN, CURRENT, &val);
			printf("[CURRENT] module power input: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_SYS_SOC, CURRENT, &val);
			printf("[CURRENT] SoC power: %.3f%s\n", convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_SYS_GPU, CURRENT, &val);
			printf("[CURRENT] GPU power rail: %.3f%s\n", convFromMilli * val,
					aunit);

			jtx1_get_ina3221(VDD_SYS_CPU, CURRENT, &val);
			printf("[CURRENT] CPU power rail: %.3f%s\n", convFromMilli * val,
					aunit);

			jtx1_get_ina3221(VDD_SYS_DDR, CURRENT, &val);
			printf("[CURRENT] (DDR) memory power rail: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_MUX, CURRENT, &val);
			printf("[CURRENT] (MUX) main carrier board power input: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_5V0_IO_SYS, CURRENT, &val);
			printf("[CURRENT] main carrier board 5V supply: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_3V3_SYS, CURRENT, &val);
			printf("[CURRENT] main carrier board 3.3V supply: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_3V3_IO_SLP, CURRENT, &val);
			printf("[CURRENT] carrier board 3.3V Sleep supply: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_1V8_IO, CURRENT, &val);
			printf("[CURRENT] main carrier board 1.8V supply: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_3V3_SYS_M2, CURRENT, &val);
			printf("[CURRENT] 3.3V supply for M.2 Key E connector: %.3f%s\n",
					convFromMilli * val, aunit);

			jtx1_get_ina3221(VDD_4V0_WIFI, CURRENT, &val);
			printf("[CURRENT] (WIFI) Antenna (?): %.3f%s\n",
					convFromMilli * val, aunit);
			printf("\n");
			//************************************************************************************************
			jtx1_get_ina3221(VDD_IN, VOLTAGE, &val);
			printf("[VOLTAGE] module power input: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_SYS_SOC, VOLTAGE, &val);
			printf("[VOLTAGE] SoC power rail: %.3f%s\n", convFromMilli * val,
					vunit);

			jtx1_get_ina3221(VDD_SYS_GPU, VOLTAGE, &val);
			printf("[VOLTAGE] GPU power rail: %.3f%s\n", convFromMilli * val,
					vunit);

			jtx1_get_ina3221(VDD_SYS_CPU, VOLTAGE, &val);
			printf("[VOLTAGE] CPU power rail: %.3f%s\n", convFromMilli * val,
					vunit);

			jtx1_get_ina3221(VDD_SYS_DDR, VOLTAGE, &val);
			printf("[VOLTAGE] (DDR) memory power rail: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_MUX, VOLTAGE, &val);
			printf("[VOLTAGE] (MUX) main carrier board power input: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_5V0_IO_SYS, VOLTAGE, &val);
			printf("[VOLTAGE] main carrier board 5V supply: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_3V3_SYS, VOLTAGE, &val);
			printf("[VOLTAGE] main carrier board 3.3V supply: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_3V3_IO_SLP, VOLTAGE, &val);
			printf("[VOLTAGE] carrier board 3.3V Sleep supply: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_1V8_IO, VOLTAGE, &val);
			printf("[VOLTAGE] main carrier board 1.8V supply: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_3V3_SYS_M2, VOLTAGE, &val);
			printf("[VOLTAGE] 3.3V supply for M.2 Key E connector: %.3f%s\n",
					convFromMilli * val, vunit);

			jtx1_get_ina3221(VDD_4V0_WIFI, VOLTAGE, &val);
			printf("[VOLTAGE] (WIFI) Antenna (?): %.3f%s\n",
					convFromMilli * val, vunit);
			printf("\n");
			//************************************************************************************************
			jtx1_get_temp(A0, &val);
			printf("[TEMPERATURE] A0: %dmC\n", val);
			jtx1_get_temp(CPU, &val);
			printf("[TEMPERATURE] CPU: %dmC\n", val);
			jtx1_get_temp(GPU, &val);
			printf("[TEMPERATURE] GPU: %dmC\n", val);
			jtx1_get_temp(PLL, &val);
			printf("[TEMPERATURE] PLL: %dmC\n", val);
			jtx1_get_temp(PMIC, &val);
			printf("[TEMPERATURE] PMIC: %dmC\n", val);
			jtx1_get_temp(TDIODE, &val);
			printf("[TEMPERATURE] TDIODE: %dmC\n", val);
			jtx1_get_temp(TBOARD, &val);
			printf("[TEMPERATURE] TBOARD: %dmC\n", val);
			jtx1_get_temp(FAN, &val);
			printf("[TEMPERATURE] FAN: %dmC\n", val);
			printf("\n");
			it_data->push_back(output);
		}
		mutex_lock.unlock();
		std::this_thread::sleep_for(std::chrono::microseconds(SLEEP_JTX2INST));
	}
}

void JTX2Inst::start_collecting_data() {
	mutex_lock.lock();
	this->data_for_iteration.clear();
	mutex_lock.unlock();

	is_locked = false;
}

void JTX2Inst::end_collecting_data() {
	mutex_lock.lock();
	is_locked = true;
	mutex_lock.unlock();
}

std::deque<std::string> JTX2Inst::get_data_from_iteration() {
	auto last = std::unique(this->data_for_iteration.begin(),
			this->data_for_iteration.end());
	this->data_for_iteration.erase(last, this->data_for_iteration.end());
	return this->data_for_iteration;
}

}
