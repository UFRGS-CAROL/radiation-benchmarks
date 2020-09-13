#include "Parameters.h"
#include "setup.h"
#include "common.h"

int main(int argc, char** argv) {
	Parameters parameters(argc, argv);
	if (parameters.verbose)
		std::cout << parameters << std::endl;

	if (parameters.use_cublas) {
		setup_gemm_cublas(parameters);
	} else if (parameters.use_cutlass) {
		throw_line("CUTLASS not ready yet!!!");
	} else  if (parameters.use_tensor_cores) {
		throw_line("Open source tensor cores not ready yet!!!");
	} else {
		if (parameters.dmr == "none") {
			setup_gemm_unhardened(parameters);
		} else {
			setup_gemm_dmr(parameters);
		}
	}

}
