#!/usr/bin/python

MAXREGS = 256

with open("src/register_kernel.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_H_\n")
    fp.write("#define REGISTER_KERNEL_H_\n")
    fp.write("\n#include "'"utils.h"'"\n")
    fp.write("\n#include "'"Parameters.h"'"\n")

    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        "\n__global__ void test_register_file_kernel(uint32 *rf, const __restrict__ uint32 *mem1, "
        "const uint64 sleep_cycles) {\n"
    )

    fp.write("\tconst uint32 i =  (blockDim.x * blockIdx.x + threadIdx.x) * RF_SIZE;\n")
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

with open("src/register_kernel_volta.h", "w") as fp:
    fp.write("#ifndef REGISTER_KERNEL_VOLTA_H_\n")
    fp.write("#define REGISTER_KERNEL_VOLTA_H_\n")
    fp.write("\n#include "'"utils.h"'"\n")
    fp.write("#include "'"Parameters.h"'"\n")

    fp.write("\n__constant__ __device__ static uint32 reg_array[RF_SIZE][2] = {\n")
    for i in range(MAXREGS):
        line_to_write = "\t\t{ 0xffffffff, 0x00000000 },\n"
        fp.write(line_to_write)

    fp.write("};\n")

    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        "\n__global__ void test_register_file_kernel_volta(uint32 *rf1, uint32 *rf2, uint32 *rf3, "
        "const uint32 zero_or_one, const uint64 sleep_cycles) {\n"
    )

    fp.write("\tconst uint32 i =  (blockDim.x * blockIdx.x + threadIdx.x) * RF_SIZE;\n")
    # fp.write("\tregister uint32 reg_errors = 0;\n")

    # setting the registers
    for i in range(0, MAXREGS):
        fp.write("\tregister uint32 r{} = __ldg(&(reg_array[{}][zero_or_one]));\n".format(i, i))

    fp.write("\n\tsleep_cuda(sleep_cycles);\n\n")

    # saving the results
    for i in range(0, MAXREGS):
        fp.write("\trf1[i + {}] = r{};\n".format(i, i))
        fp.write("\trf2[i + {}] = r{};\n".format(i, i))
        fp.write("\trf3[i + {}] = r{};\n\n".format(i, i))

    fp.write("}\n\n")
    fp.write("#endif /* REGISTER_KERNEL_VOLTA_H_ */\n")
