

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "common.h"


int main(int argc, char** argv) {
	int num_devices;
	cudaGetDeviceCount(&num_devices);
	if (num_devices > 1)
		cudaSetDevice(DEVICE);

	run(argc, argv);

	return EXIT_SUCCESS;
}
