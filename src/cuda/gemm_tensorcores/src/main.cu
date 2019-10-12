#include <string>
#include "Log.h"
#include "common.h"

void call_mxm_unhardened(Log& log_obj, GEMMTYPE gemm_t);
void call_mxm_dmr(Log& log_obj, GEMMTYPE gemm_t);

void usage(char **argv) {
	std::cout << "./" << argv[0]
			<< " --generate 0/1 --gold <gold file, DEFAULT=./gold.matrix > --size <matrix size, DEFAULT=8192> "
					"--iterations <how many iterations, optional> --input_a <input A, DEFAUL=./input_a.matrix> "
					"--input_b <input B, DEFAUL=./input_b.matrix> --input_c <input C, DEFAUL=./input_c.matrix>  --precision <float/double, DEFAULT=float>"
			<< std::endl;
}

int main(int argc, char** argv) {
	Log log_obj(argc, argv);
	std::cout << log_obj << std::endl;

	//DMR TYPES
	if (log_obj.dmr == "dmr") {
		if (log_obj.use_tensor_cores) {
			call_mxm_dmr(log_obj, NONDMRWMMA);
		} else {
			call_mxm_dmr(log_obj, NONDMRGEMM);
		}

	} else if (log_obj.dmr == "dmrmixed") {

	} else if (log_obj.dmr == "nondmr") {
		if (log_obj.use_tensor_cores) {
			call_mxm_unhardened(log_obj, NONDMRWMMA);
		} else {
			call_mxm_unhardened(log_obj, NONDMRGEMM);
		}
	}
	std::cout << "Finished computation\n";
	return 0;
}
