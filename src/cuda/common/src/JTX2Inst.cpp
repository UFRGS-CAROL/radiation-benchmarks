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
#include <sstream>
#include <iomanip>      // std::setprecision
#include <fstream>

#include "JTX2Inst.h"

#include "jtx2/include/jtx1inst.h"

namespace rad {


JTX2Inst::JTX2Inst(unsigned device_index, std::string& output_file) : Profiler(device_index, output_file){
	this->_thread_profiler = std::thread(JTX2Inst::data_colector,
			&this->_output_log_file, &this->_thread_running, &this->_is_locked);
}

JTX2Inst::~JTX2Inst() {
	this->_thread_running = false;
	this->_thread_profiler.join();
}

void JTX2Inst::data_colector(std::string* output_log_file, std::atomic<bool>* _thread_running,
		std::atomic<bool>* _is_locked) {
	unsigned int val;
//	unsigned long rate;
	float convFromMilli;
	std::string wunit, aunit, vunit;
	int convert;

	convert = 0;

	if (convert) {
		convFromMilli = 0.001;
		wunit = "W";
		aunit = "A";
		vunit = "V";
	} else {
		convFromMilli = 1.0;
		wunit = "mW";
		aunit = "mA";
		vunit = "mV";
	}

	std::ofstream out_stream(*output_log_file);
	if (out_stream.good() == false) {
		std::runtime_error(*output_log_file + " NOT GOOD FOR WRITING");
	}
	out_stream << "POWER UNIT:" << wunit << ";CURRENT UNIT:" << aunit
			<< ";VOLTAGE UNIT:" << vunit << ";convFromMili:" << convFromMilli
			<< std::endl;
	out_stream
			<< "TIMESTAMP;VDD_IN_POWER;VDD_SYS_SOC_POWER;VDD_SYS_GPU_POWER;VDD_SYS_CPU_POWER;"
					"VDD_SYS_DDR_POWER;VDD_MUX_POWER;VDD_5V0_IO_SYS_POWER;VDD_3V3_SYS_POWER;"
					"VDD_3V3_IO_SLP_POWER;VDD_1V8_IO_POWER;VDD_3V3_SYS_M2_POWER;VDD_4V0_WIFI_POWER;"
					"VDD_IN_CURRENT;VDD_SYS_SOC_CURRENT;VDD_SYS_GPU_CURRENT;VDD_SYS_CPU_CURRENT;"
					"VDD_SYS_DDR_CURRENT;VDD_MUX_CURRENT;VDD_5V0_IO_SYS_CURRENT;VDD_3V3_SYS_CURRENT;"
					"VDD_3V3_IO_SLP_CURRENT;VDD_1V8_IO_CURRENT;VDD_3V3_SYS_M2_CURRENT;VDD_4V0_WIFI_CURRENT;"
					"VDD_IN_VOLTAGE;VDD_SYS_SOC_VOLTAGE;VDD_SYS_GPU_VOLTAGE;VDD_SYS_CPU_VOLTAGE;"
					"VDD_SYS_DDR_VOLTAGE;VDD_MUX_VOLTAGE;VDD_5V0_IO_SYS_VOLTAGE;VDD_3V3_SYS_VOLTAGE;"
					"VDD_3V3_IO_SLP_VOLTAGE;VDD_1V8_IO_VOLTAGE;VDD_3V3_SYS_M2_VOLTAGE;VDD_4V0_WIFI_VOLTAGE;"
					"A0_TEMPERATURE;CPU_TEMPERATURE;GPU_TEMPERATURE;PLL_TEMPERATURE;PMIC_TEMPERATURE;"
					"TDIODE_TEMPERATURE;TBOARD_TEMPERATURE;FAN_TEMPERATURE"
			<< std::endl;
	while (*_thread_running) {
		if (*_is_locked) {
			std::time_t result = std::time(nullptr);
			std::string asc_time(std::asctime(std::localtime(&result)));
			asc_time.pop_back();
			out_stream << "[" << asc_time << "];";

//			out_stream << std::scientific << std::setprecision(7);
			//***********************************************************************************************
			jtx1_get_ina3221(VDD_IN, POWER, &val);
//			printf("[POWER] module power input: %.3f%s\n", convFromMilli * val,
//							wunit);
			float power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_SYS_SOC, POWER, &val);
//			printf("[POWER] SoC power: %.3f%s\n", convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_SYS_GPU, POWER, &val);
//			printf("[POWER] GPU power rail: %.3f%s\n", convFromMilli * val,
//					wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_SYS_CPU, POWER, &val);
//			printf("[POWER] CPU power rail: %.3f%s\n", convFromMilli * val,
//					wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_SYS_DDR, POWER, &val);
//			printf("[POWER] (DDR) memory power rail: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_MUX, POWER, &val);
//			printf("[POWER] (MUX) main carrier board power input: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_5V0_IO_SYS, POWER, &val);
//			printf("[POWER] main carrier board 5V supply: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS, POWER, &val);
//			printf("[POWER] main carrier board 3.3V supply: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_3V3_IO_SLP, POWER, &val);
//			printf("[POWER] carrier board 3.3V Sleep supply: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_1V8_IO, POWER, &val);
//			printf("[POWER] main carrier board 1.8V supply: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS_M2, POWER, &val);
//			printf("[POWER] 3.3V supply for M.2 Key E connector: %.3f%s\n",
//					convFromMilli * val, wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

			jtx1_get_ina3221(VDD_4V0_WIFI, POWER, &val);
//			printf("[POWER] (WIFI) Antenna (?): %.3f%s\n", convFromMilli * val,
//					wunit);
			power_out = convFromMilli * val;
			out_stream << power_out << ";";

//			printf("\n");

//************************************************************************************************
			jtx1_get_ina3221(VDD_IN, CURRENT, &val);
//			printf("[CURRENT] module power input: %.3f%s\n",
//					convFromMilli * val, aunit);
			float current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_SYS_SOC, CURRENT, &val);
//			printf("[CURRENT] SoC power: %.3f%s\n", convFromMilli * val, aunit);
			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_SYS_GPU, CURRENT, &val);
//			printf("[CURRENT] GPU power rail: %.3f%s\n", convFromMilli * val,
//					aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_SYS_CPU, CURRENT, &val);
//			printf("[CURRENT] CPU power rail: %.3f%s\n", convFromMilli * val,
//					aunit);
			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_SYS_DDR, CURRENT, &val);
//			printf("[CURRENT] (DDR) memory power rail: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_MUX, CURRENT, &val);
//			printf("[CURRENT] (MUX) main carrier board power input: %.3f%s\n",
//					convFromMilli * val, aunit);
			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_5V0_IO_SYS, CURRENT, &val);
//			printf("[CURRENT] main carrier board 5V supply: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS, CURRENT, &val);
//			printf("[CURRENT] main carrier board 3.3V supply: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_3V3_IO_SLP, CURRENT, &val);
//			printf("[CURRENT] carrier board 3.3V Sleep supply: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_1V8_IO, CURRENT, &val);
//			printf("[CURRENT] main carrier board 1.8V supply: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS_M2, CURRENT, &val);
//			printf("[CURRENT] 3.3V supply for M.2 Key E connector: %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";

			jtx1_get_ina3221(VDD_4V0_WIFI, CURRENT, &val);
//			printf("[CURRENT] (WIFI) Antenna (?): %.3f%s\n",
//					convFromMilli * val, aunit);

			current_out = convFromMilli * val;
			out_stream << current_out << ";";
//			printf("\n");
			//************************************************************************************************
			jtx1_get_ina3221(VDD_IN, VOLTAGE, &val);
//			printf("[VOLTAGE] module power input: %.3f%s\n",
//					convFromMilli * val, vunit);

			float voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_SYS_SOC, VOLTAGE, &val);
//			printf("[VOLTAGE] SoC power rail: %.3f%s\n", convFromMilli * val,
//					vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_SYS_GPU, VOLTAGE, &val);
//			printf("[VOLTAGE] GPU power rail: %.3f%s\n", convFromMilli * val,
//					vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_SYS_CPU, VOLTAGE, &val);
//			printf("[VOLTAGE] CPU power rail: %.3f%s\n", convFromMilli * val,
//					vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_SYS_DDR, VOLTAGE, &val);
//			printf("[VOLTAGE] (DDR) memory power rail: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_MUX, VOLTAGE, &val);
//			printf("[VOLTAGE] (MUX) main carrier board power input: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_5V0_IO_SYS, VOLTAGE, &val);
//			printf("[VOLTAGE] main carrier board 5V supply: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS, VOLTAGE, &val);
//			printf("[VOLTAGE] main carrier board 3.3V supply: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_3V3_IO_SLP, VOLTAGE, &val);
//			printf("[VOLTAGE] carrier board 3.3V Sleep supply: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_1V8_IO, VOLTAGE, &val);
//			printf("[VOLTAGE] main carrier board 1.8V supply: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_3V3_SYS_M2, VOLTAGE, &val);
//			printf("[VOLTAGE] 3.3V supply for M.2 Key E connector: %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";

			jtx1_get_ina3221(VDD_4V0_WIFI, VOLTAGE, &val);
//			printf("[VOLTAGE] (WIFI) Antenna (?): %.3f%s\n",
//					convFromMilli * val, vunit);

			voltage_out = convFromMilli * val;
			out_stream << voltage_out << ";";
//			printf("\n");
			//************************************************************************************************
			jtx1_get_temp(A0, &val);
//			printf("[TEMPERATURE] A0: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(CPU, &val);
//			printf("[TEMPERATURE] CPU: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(GPU, &val);
//			printf("[TEMPERATURE] GPU: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(PLL, &val);
//			printf("[TEMPERATURE] PLL: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(PMIC, &val);
//			printf("[TEMPERATURE] PMIC: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(TDIODE, &val);
//			printf("[TEMPERATURE] TDIODE: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(TBOARD, &val);
//			printf("[TEMPERATURE] TBOARD: %dmC\n", val);
			out_stream << val << ";";

			jtx1_get_temp(FAN, &val);
//			printf("[TEMPERATURE] FAN: %dmC\n", val);
//			printf("\n");
			out_stream << val << std::endl;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(PROFILER_SLEEP));
	}

	out_stream.close();
}

void JTX2Inst::start_profile() {
//	mutex_lock.lock();
//	this->data_for_iteration.clear();
//	mutex_lock.unlock();
//
//	is_locked = false;
	this->_is_locked = true;
}

void JTX2Inst::end_profile() {
//	mutex_lock.lock();
//	is_locked = true;
//	mutex_lock.unlock();
	this->_is_locked = false;
}
}
