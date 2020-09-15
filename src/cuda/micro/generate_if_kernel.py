#!/usr/bin/python

MAXBRANCHS = 1024

with open("src/branch_kernel.h", "w") as fp:
    fp.write("#ifndef BRANCH_KERNEL_H_\n")
    fp.write("#define BRANCH_KERNEL_H_\n\n")
    fp.write("#include <cstdint>\n\n")


    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        "\ntemplate<typename int_t>"
        "\n__global__ void int_branch_kernel(int_t* dst_1, int_t* dst_2, int_t* dst_3, uint32_t op) {\n"
    )

    fp.write("\tconst int_t i = (blockDim.x * blockIdx.x + threadIdx.x);\n")
    fp.write("\tint_t value = i;\n")
    fp.write("\tint_t to_store = (i + 1) % op;\n")
    fp.write("\tfor(int opi = 0; opi < op; opi++) {")

    fp.write("\n\t\tif (threadIdx.x == 0) {\n"
             "\t\t\tvalue = to_store;\n"
             "\t\t}")

    # setting the registers
    for i in range(1, MAXBRANCHS):
        fp.write(" else if (threadIdx.x == {})".format(i) + " {\n")  # " dst[i] = {};".format(i, i))
        fp.write("\t\t\tvalue = to_store;\n")
        fp.write("\t\t}")

    fp.write("\n\t}")
    fp.write("\n\tdst_1[i] = value;\n")
    fp.write("\n\tdst_2[i] = value;\n")
    fp.write("\n\tdst_3[i] = value;\n")

    fp.write("\n}\n\n")
    fp.write("#endif /* BRANCH_KERNEL_H_ */\n")
