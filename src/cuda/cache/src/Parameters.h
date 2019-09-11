/*
 * Parameters.h
 *
 *  Created on: Sep 3, 2019
 *      Author: carol
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "Log.h"
#include <unordered_map>

#define BLOCK_SIZE 32

/**
 * It was 256 registers per threads when this program was created
 */
#if __CUDA_ARCH__ <= 750

//CACHE LINE PARAMETERS TO TEST RF, L1, and L2
#define RF_SIZE 256
#define CACHE_LINE_SIZE 128U
#define CACHE_LINE_SIZE_BY_INT32 CACHE_LINE_SIZE/ 8

//SHARED MEMORY PARAMETERS TO FORCE THE BLOCKS EXECUTE
//WHITHIN A SM
#define MAX_VOLTA_SHARED_MEMORY_TO_TEST_L1 2 * 1024
#define MAX_KEPLER_SHARED_MEMORY_TO_TEST_L1 1 * 1024

//SHARED MEMORY PARAMETERS TO TEST SHARED MEMORY
#define MAX_VOLTA_SHARED_MEMORY 48 * 1024
#define MAX_KEPLER_SHARED_MEMORY 48 * 1024

//KEPLER L1 MEMORY PARAMETERS
#define MAX_VOLTA_L1_MEMORY 48 * 1024
#define MAX_KEPLER_L1_MEMORY 48 * 1024

//READ ONLY PARAMETERS FOR KEPLER
#define MAX_VOLTA_CONSTANT_MEMORY 64 * 1024
#define MAX_KEPLER_CONSTANT_MEMORY 64 * 1024


#define BLOCK_PER_SM 1
#endif

#define DEVICE_INDEX 0 //Radiation test can be done only one device at time

typedef enum {
	K20, K40, TEGRAX2, XAVIER, TITANV, BOARD_COUNT,
} Board;

struct Parameters {
	uint32 number_of_sms;
	Board device;
	uint32 shared_memory_size;
	uint32 l2_size;
	uint64 one_second_cycles; // the necessary number of cycles to count one second

	std::string board_name;

	//register file size
	uint32 registers_per_block;

	//const memory
	uint32 const_memory_per_block;

	uint64 clock_rate;

	std::unordered_map<std::string, Board> devices_name = {
	//Tesla K20
			{ "Tesla K20c", K20 },
			//Tesla K40
			{ "Tesla K40c", K40 },
			// Titan V
			{ "TITAN V", TITANV },
			//Xavier
			{ "Xavier", XAVIER }
	//Other
			};

	Parameters() {

		auto device_info = get_device_information(DEVICE_INDEX);
		std::string device_name(device_info.name);
		if (devices_name.find(device_name) == devices_name.end())
			error("CANNOT FOUND THE DEVICE\n");

		this->device = devices_name[device_name];
		this->number_of_sms = device_info.multiProcessorCount;
		this->shared_memory_size = device_info.sharedMemPerMultiprocessor;
		this->l2_size = device_info.l2CacheSize;
		this->board_name = device_info.name;
		this->registers_per_block = device_info.regsPerBlock;
		this->const_memory_per_block = device_info.totalConstMem;
		this->clock_rate = device_info.clockRate;
	}

	void set_setup_sleep_time(const uint64 seconds_sleep) {
		this->one_second_cycles = clock_rate * 1000 * seconds_sleep;
	}

	friend std::ostream& operator<<(std::ostream& os, const Parameters& par) {
		os << "Testing " << par.board_name;
		os << " GPU. Using " << par.number_of_sms;
		os << "SMs, one second cycles " << par.one_second_cycles;
		return os;
	}

	cudaDeviceProp get_device_information(int dev) {
		int driver_version, runtime_version;
		cudaSetDevice(dev);
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, dev);

		std::printf("Radiation test on device %d: \"%s\"\n", dev,
				device_prop.name);

		// Console log
		cudaDriverGetVersion(&driver_version);
		cudaRuntimeGetVersion(&runtime_version);
		std::printf(
				"  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
				driver_version / 1000, (driver_version % 100) / 10,
				runtime_version / 1000, (runtime_version % 100) / 10);
		std::printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
				device_prop.major, device_prop.minor);

		std::printf(
				"  Total amount of global memory:                 %.0f MBytes "
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
		std::printf(
				"  Memory Clock rate:                             %.0f Mhz\n",
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

		std::printf(
				"  Total amount of constant memory:               %lu bytes\n",
				device_prop.totalConstMem);
		std::printf(
				"  Total amount of shared memory per block:       %lu bytes\n",
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
		std::printf(
				"  Maximum memory pitch:                          %lu bytes\n",
				device_prop.memPitch);
		std::printf(
				"  Texture alignment:                             %lu bytes\n",
				device_prop.textureAlignment);
		std::printf(
				"  Concurrent copy and kernel execution:          %s with %d copy "
						"engine(s)\n",
				(device_prop.deviceOverlap ? "Yes" : "No"),
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
		//	std::printf("  Device supports Compute Preemption:            %s\n",
		//			device_prop.computePreemptionSupported ? "Yes" : "No");
		//	std::printf("  Supports Cooperative Kernel Launch:            %s\n",
		//			device_prop.cooperativeLaunch ? "Yes" : "No");
		//	std::printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
		//			device_prop.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		std::printf(
				"  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
				device_prop.pciDomainID, device_prop.pciBusID,
				device_prop.pciDeviceID);
		return device_prop;
	}
};

#endif /* PARAMETERS_H_ */
