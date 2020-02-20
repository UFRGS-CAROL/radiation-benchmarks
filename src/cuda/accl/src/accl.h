#ifndef SCCL_CU_H
#define SCCL_CU_H
double acclCuda(std::vector<int>& out, std::vector<int>& components,
		const std::vector<int>& in, uint nFrames, uint nFramsPerStream,
		const int rows, const int cols, int logs_active);
#endif
