#include "Log.h"
#include "setup.h"
#include "common.h"

void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --gold <gold file, DEFAULT=./gold.matrix > --size <matrix size, DEFAULT=8192> "
					"--iterations <how many iterations, optional> --input_a <input A, DEFAUL=./input_a.matrix> "
					"--input_b <input B, DEFAUL=./input_b.matrix> --input_c <input C, DEFAUL=./input_c.matrix>  --precision <float/double, DEFAULT=float>"
			<< std::endl;
}

int main(int argc, char** argv) {
	Log log(argc, argv);
	std::cout << log << std::endl;

	if (log.use_tensor_cores) {
		if (log.dmr == "none") {
			setup_gemm_tensor_cores_unhardened(log);
		} else {
			setup_gemm_tensor_cores_dmr(log);
		}
	} else {
		if (log.dmr == "none") {
			setup_gemm_unhardened(log);
		} else {
			setup_gemm_dmr(log);
		}
	}

}
