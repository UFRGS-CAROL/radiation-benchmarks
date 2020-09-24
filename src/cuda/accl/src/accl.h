#ifndef SCCL_CU_H
#define SCCL_CU_H
//double acclCuda(std::vector<int>& out, std::vector<int>& components,
//		const std::vector<int>& in, uint nFrames, uint nFramsPerStream,
//		const int rows, const int cols, int logs_active);

double acclCuda(rad::DeviceVector<int>& devOut,
		rad::DeviceVector<int>& devComponents,
		const rad::DeviceVector<int>& devIn, uint nFrames, uint nFramsPerStream,
		const int rows, const int cols, int logs_active, rad::Log& log,
		std::vector<cudaStream_t>& streams);

std::string get_multi_compiler_header();
#endif
