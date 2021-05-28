#include "Parameters.h"
#include "common.h"

/**
 * Setup for common MxM (GEMM)
 */
void setup_gemm_unhardened(Parameters&);
void setup_gemm_dmr(Parameters&);
void setup_gemm_cublas(Parameters&);

/**
 * Setup for Tensor (GEMM)
 */
void setup_gemm_tensor_cores_unhardened(Parameters&);
void setup_gemm_tensor_cores_dmr(Parameters&);

/**
 * Get the __CUDACC_VER_MAJOR__ from NVCC
 */

int main(int argc, char** argv) {
	std::cout << std::boolalpha;

	Parameters parameters(argc, argv);
	if (parameters.verbose)
		std::cout << parameters << std::endl;

	if (parameters.use_cublas) {
		setup_gemm_cublas(parameters);
	} else if (parameters.use_cutlass) {
		throw_line("CUTLASS not ready yet!!!");
	} else if (parameters.use_tensor_cores) {
		if (parameters.dmr == "none") {
			setup_gemm_tensor_cores_unhardened(parameters);
		} else {
			setup_gemm_tensor_cores_dmr(parameters);
		}
	} else {
		if (parameters.dmr == "none") {
			setup_gemm_unhardened(parameters);
		} else {
			setup_gemm_dmr(parameters);
		}
	}

}
