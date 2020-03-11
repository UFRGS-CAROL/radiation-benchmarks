#!/usr/bin/python

MAXBRANCHS = 1024

with open("src/branch_kernel.h", "w") as fp:
    fp.write("#ifndef BRANCH_KERNEL_H_\n")
    fp.write("#define BRANCH_KERNEL_H_\n")


    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        "\ntemplate<uint32_t UNROLL_MAX, typename int_t>"
        "\n__global__ void branch_int_kernel(int_t* src, int_t* dst, uint32_t op) {\n"
    )

    fp.write("\tconst uint32_t i =  (blockDim.x * blockIdx.x + threadIdx.x);\n")

    fp.write("\n\tif (threadIdx.x == 0) dst[i] = 0;")

    # setting the registers
    for i in range(1, MAXBRANCHS):
        fp.write("\n\telse if (threadIdx.x == {})".format(i) + "{\n")  # " dst[i] = {};".format(i, i))
        fp.write("\t\tdst[i] = threadIdx.x + ((threadIdx.x % 2) ?  -UNROLL_MAX : UNROLL_MAX) + {};\n".format(i))
        fp.write("\t}")



    fp.write("\n}\n\n")
    fp.write("#endif /* BRANCH_KERNEL_H_ */\n")
