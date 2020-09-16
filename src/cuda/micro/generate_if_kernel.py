#!/usr/bin/python

MAXBRANCHS = 1024


def is_prime(num):
    if num > 1:
        # check for factors
        for ii in range(2, num):
            if (num % ii) == 0:
                return False
        else:
            return True

    # if input number is less than
    # or equal to 1, it is not prime
    else:
        return False


with open("src/branch_kernel.cu", "w") as fp:
    # fp.write("#ifndef BRANCH_KERNEL_H_\n")
    # fp.write("#define BRANCH_KERNEL_H_\n\n")
    fp.write("#include <cstdint>\n")
    fp.write("#include \"input_device.h\"\n\n")
    fp.write("#include \"branch_kernel.h\"\n\n")

    ####################################################################
    # kernel for FFFF which must be or
    fp.write(
        # "\ntemplate<typename int_t>"
        "\n__global__ void int_branch_kernel(int* dst_1, int* dst_2, int* dst_3, unsigned op) {\n"
    )

    fp.write("\tconst unsigned i = (blockDim.x * blockIdx.x + threadIdx.x);\n")
    fp.write("\tint value = 0;\n")

    fp.write("\tfor(unsigned opi = 0; opi < op; opi++) {")

    fp.write("\n\t\tif (threadIdx.x == 0) {\n"
             "\t\t\tvalue = common_int_input[threadIdx.x] & opi;\n"
             "\t\t}")

    # setting the registers
    for i in range(1, MAXBRANCHS):
        fp.write(" else if (threadIdx.x == {})".format(i) + " {\n")
        # if is_prime(i) and i > 900:
        #     fp.write("\t\t\tvalue = common_int_input[threadIdx.x] & opi;\n")
        # else:
        fp.write("\t\t\tvalue = common_int_input[threadIdx.x];\n")

        fp.write("\t\t}")

    fp.write("\n\t}")
    fp.write("\n\tdst_1[i] = value;\n")
    fp.write("\n\tdst_2[i] = value;\n")
    fp.write("\n\tdst_3[i] = value;\n")

    fp.write("\n}\n\n")
    # fp.write("#endif /* BRANCH_KERNEL_H_ */\n")
