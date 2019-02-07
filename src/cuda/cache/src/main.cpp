/*
 ============================================================================
 Name        : main.cpp
 Author      : Fernando
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <unordered_map>

#include "NVMLWrapper.h"
#include "kernels.h"
#include "utils.h"
#include "Log.h"
//#include "CacheProfiler.h"

#define DEVICE_INDEX 0 //Radiation test can be done only one device at time

cudaDeviceProp get_device_information(int dev) {
	int driver_version, runtime_version;
	cudaSetDevice(dev);
	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, dev);

	std::printf("Radiation test on device %d: \"%s\"\n", dev, device_prop.name);

	// Console log
	cudaDriverGetVersion(&driver_version);
	cudaRuntimeGetVersion(&runtime_version);
	std::printf(
			"  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driver_version / 1000, (driver_version % 100) / 10,
			runtime_version / 1000, (runtime_version % 100) / 10);
	std::printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			device_prop.major, device_prop.minor);

	std::printf("  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(device_prop.totalGlobalMem / 1048576.0f),
			(unsigned long long) device_prop.totalGlobalMem);

	std::printf(
			"  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			device_prop.multiProcessorCount,
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor),
			_ConvertSMVer2Cores(device_prop.major, device_prop.minor)
					* device_prop.multiProcessorCount);
	std::printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
					"GHz)\n", device_prop.clockRate * 1e-3f,
			device_prop.clockRate * 1e-6f);

	// This is supported in CUDA 5.0 (runtime API device properties)
	std::printf("  Memory Clock rate:                             %.0f Mhz\n",
			device_prop.memoryClockRate * 1e-3f);
	std::printf("  Memory Bus Width:                              %d-bit\n",
			device_prop.memoryBusWidth);

	if (device_prop.l2CacheSize) {
		std::printf(
				"  L2 Cache Size:                                 %d bytes\n",
				device_prop.l2CacheSize);
	}

	std::printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
					"%d), 3D=(%d, %d, %d)\n", device_prop.maxTexture1D,
			device_prop.maxTexture2D[0], device_prop.maxTexture2D[1],
			device_prop.maxTexture3D[0], device_prop.maxTexture3D[1],
			device_prop.maxTexture3D[2]);
	std::printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			device_prop.maxTexture1DLayered[0],
			device_prop.maxTexture1DLayered[1]);
	std::printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
					"layers\n", device_prop.maxTexture2DLayered[0],
			device_prop.maxTexture2DLayered[1],
			device_prop.maxTexture2DLayered[2]);

	std::printf("  Total amount of constant memory:               %lu bytes\n",
			device_prop.totalConstMem);
	std::printf("  Total amount of shared memory per block:       %lu bytes\n",
			device_prop.sharedMemPerBlock);
	std::printf("  Total number of registers available per block: %d\n",
			device_prop.regsPerBlock);
	std::printf("  Warp size:                                     %d\n",
			device_prop.warpSize);
	std::printf("  Maximum number of threads per multiprocessor:  %d\n",
			device_prop.maxThreadsPerMultiProcessor);
	std::printf("  Maximum number of threads per block:           %d\n",
			device_prop.maxThreadsPerBlock);
	std::printf(
			"  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
			device_prop.maxThreadsDim[2]);
	std::printf(
			"  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			device_prop.maxGridSize[0], device_prop.maxGridSize[1],
			device_prop.maxGridSize[2]);
	std::printf("  Maximum memory pitch:                          %lu bytes\n",
			device_prop.memPitch);
	std::printf("  Texture alignment:                             %lu bytes\n",
			device_prop.textureAlignment);
	std::printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
					"engine(s)\n", (device_prop.deviceOverlap ? "Yes" : "No"),
			device_prop.asyncEngineCount);
	std::printf("  Run time limit on kernels:                     %s\n",
			device_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
	std::printf("  Integrated GPU sharing Host Memory:            %s\n",
			device_prop.integrated ? "Yes" : "No");
	std::printf("  Support host page-locked memory mapping:       %s\n",
			device_prop.canMapHostMemory ? "Yes" : "No");
	std::printf("  Alignment requirement for Surfaces:            %s\n",
			device_prop.surfaceAlignment ? "Yes" : "No");
	std::printf("  Device has ECC support:                        %s\n",
			device_prop.ECCEnabled ? "Enabled" : "Disabled");
	std::printf("  Device supports Unified Addressing (UVA):      %s\n",
			device_prop.unifiedAddressing ? "Yes" : "No");
	std::printf("  Device supports Compute Preemption:            %s\n",
			device_prop.computePreemptionSupported ? "Yes" : "No");
	std::printf("  Supports Cooperative Kernel Launch:            %s\n",
			device_prop.cooperativeLaunch ? "Yes" : "No");
	std::printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
	std::printf(
			"  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			device_prop.pciDomainID, device_prop.pciBusID,
			device_prop.pciDeviceID);
	return device_prop;
}

void set_cache_config(const std::string memory) {
	if (memory == "L1") {
		cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
//		cuda_check(cudaThreadSetCacheConfig(cudaFuncCachePreferL1));

	} else if (memory == "SHARED") {
		cuda_check(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
//		cuda_check(cudaThreadSetCacheConfig(cudaFuncCachePreferShared));
	}
}

void compare(const Tuple& t, Log& log, const byte gold_byte) {
	//Checking the misses
	uint32 hits = 0;
	uint32 misses = t.misses.size();
	uint32 false_hit = 0;
	for (uint32 i = 0; i < t.hits.size(); i++) {
		int32 hit = t.hits[i];
		int32 miss = t.misses[i];
		if (hit < miss) {
			hits++;
		}
		if (miss < hit) {
			false_hit++;
		}
	}

	//Checking the errors
	for (uint32 i = 0; i < t.cache_lines.size(); i++) {
		auto found_byte = t.cache_lines[i];

		if (found_byte != gold_byte) {
			auto cache_line = i / 128; //supposing that all lines have 128 bytes

			std::string error_detail = "";

			error_detail += " i:" + std::to_string(i);
			error_detail += " cache_line:" + std::to_string(cache_line);
			error_detail += " e:" + std::to_string(gold_byte);
			error_detail += " r:" + std::to_string(found_byte);

			//log error detail already increment error var
			log.log_error(error_detail);
		}
	}

	if (log.errors != t.errors) {
		std::string error_detail = "errors on the data path. expected:"
				+ std::to_string(t.errors) + " found:"
				+ std::to_string(log.errors);
		log.log_error(error_detail);
	}
}

int main(int argc, char **argv) {
	std::unordered_map<std::string, Board> devices_name = {
	//Tesla K40
			{ "Tesla K40c", K40 },
			// Titan V
			{ "TITAN V", TITANV }
	//Other
			};

	//Const list of NVIDIA DEVICES
	bool l2_checked = false;
#ifdef L2TEST
	l2_checked = true;
#endif
	auto device_info = get_device_information(DEVICE_INDEX);
	std::string device_name(device_info.name);
	if (devices_name.find(device_name) == devices_name.end())
		error("CANNOT FOUND THE DEVICE\n");

	//Parameter to the functions
	Parameters test_parameter;
	test_parameter.device = devices_name[device_name];
	test_parameter.number_of_sms = device_info.multiProcessorCount;
	test_parameter.shared_memory_size = device_info.sharedMemPerMultiprocessor;
	test_parameter.l2_size = device_info.l2CacheSize;
	test_parameter.one_second_cycles = device_info.clockRate * 1000;
	test_parameter.board_name = device_info.name;

	//Log obj
	Log log(argc, argv, device_name, test_parameter.shared_memory_size,
			test_parameter.l2_size, test_parameter.number_of_sms,
			test_parameter.one_second_cycles);
	log.set_info_max(2000);

	test_parameter.log = &log;

	/**
	 * SETUP THE NVWL THREAD
	 */
	NVMLWrapper counter_thread(DEVICE_INDEX);
	std::cout << "Testing " << test_parameter.board_name << " GPU. Using "
			<< test_parameter.number_of_sms << "SMs, one second cycles "
			<< test_parameter.one_second_cycles << std::endl;

	for (uint64 iteration = 0; iteration < log.iterations;) {
		//set memory config
		set_cache_config(log.test_mode);

		for (byte t_byte : { 0xff, 0x00 }) {
			//Start collecting data
			counter_thread.start_collecting_data();
			test_parameter.t_byte = t_byte;

			double start_it = log.mysecond();
			//Start iteration
			log.start_iteration_app();

			Tuple ret;

			//test L1
			if (log.test_mode == "L1") {
				ret = test_l1_cache(test_parameter);
			}
			//Test l2
			if (log.test_mode == "L2") {
				if (l2_checked == false) {
					error(
							"YOU MUST BUILD CUDA CACHE TEST WITH: make DISABLEL1CACHE=1");
				}
				ret = test_l2_cache(test_parameter);
			}
			//Test Shared
			if (log.test_mode == "SHARED") {
				ret = test_shared_memory(test_parameter);
			}
			//Test Constant
			if (log.test_mode == "CONSTANT") {
				ret = test_read_only_cache(test_parameter);
			}
			//Test Registers
			if (log.test_mode == "REGISTERS") {
				ret = test_register_file(test_parameter);
			}

			//end iteration
			log.end_iteration_app();
			double end_it = log.mysecond();

			//End collecting the data
			counter_thread.end_collecting_data();

			double start_dev_reset = log.mysecond();
			//reset the device
			cuda_check(cudaDeviceReset());
			double end_dev_reset = log.mysecond();

			//Comparing the output
			double start_cmp = log.mysecond();
			compare(ret, log, test_parameter.t_byte);
			double end_cmp = log.mysecond();
			//update errors
			if (log.errors) {
				log.update_error_count();

				auto iteration_data = counter_thread.get_data_from_iteration();
				for (auto info_line : iteration_data) {
					log.log_info(info_line);
				}

			}

			std::cout << "Iteration: " << iteration << " Time: "
					<< end_it - start_it << " Errors: " << log.errors
					<< " Device Reset: " << end_dev_reset - start_dev_reset
					<< " Comparing: " << end_cmp - start_cmp << std::endl;

			iteration++;
		}
	}
	return 0;

}

