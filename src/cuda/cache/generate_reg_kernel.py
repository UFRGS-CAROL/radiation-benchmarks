#!/usr/bin/python

MAXREGS = 256

with open("src/register_kernel.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_H_\n")
    fp.write("#define REGISTER_KERNEL_H_\n")
    fp.write("\n#include "'"utils.h"'"\n")

    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        "\n__global__ void test_register_file_kernel(uint32 *rf, const __restrict__ uint32 *mem1, "
        "const uint64 sleep_cycles) {\n"
    )

    fp.write("\tconst uint32 i =  (blockDim.x * blockIdx.x + threadIdx.x) * 256;\n")
    # fp.write("\tregister uint32 reg_errors = 0;\n")

    # setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = __ldg(mem1 + {});\n".format(i, i))

    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")

    # saving the results
    for i in range(0, MAXREGS):
        fp.write("\trf[i + {}] = r{};\n".format(i, i))

    fp.write("}\n\n")
    fp.write("#endif /* REGISTER_KERNEL_H_ */\n")
