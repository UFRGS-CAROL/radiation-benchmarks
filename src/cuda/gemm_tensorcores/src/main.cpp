#include "Parameters.h"
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
    Parameters parameters(argc, argv);
    if (parameters.verbose)
        std::cout << parameters << std::endl;

    if (parameters.use_tensor_cores && !parameters.use_cublas) {
        throw_line("Open source tensor cores not ready yet!!!");
    } else if (parameters.use_cublas){
        setup_gemm_cublas(parameters);
    }else if (parameters.use_cutlass){

    } else {
        if (parameters.dmr == "none") {
            setup_gemm_unhardened(parameters);
        } else {
            setup_gemm_dmr(parameters);
        }
    }

}
