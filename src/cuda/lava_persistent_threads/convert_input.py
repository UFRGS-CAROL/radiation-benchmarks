#!/usr/bin/python

import struct
import getopt
import sys

BLOCK_SIZE = 4


def load_binary_file(file_path, n):
    with open(file_path, "rb") as file:
        file_content = struct.unpack('f' * n, file.read(BLOCK_SIZE * n))

    return file_content


def generate_input_h(file_path, input_size, output_file, var_name):
    with open(output_file, "w") as fp:
        file_content = load_binary_file(file_path, input_size)
        header = "#ifndef {}\n#define {}\n".format(output_file.replace(".h", "_H").upper(),
                                                   output_file.replace(".h", "_H").upper())
        # header += "\n#include <vector>\n\n"
        header += "#ifndef ARRAY_SIZE\n"
        header += "#define ARRAY_SIZE {}\n".format(input_size)
        header += "#endif\n\n"
        header += "__device__ float {}[ARRAY_SIZE] = ".format(var_name) + "{\n"

        for i in range(0, len(file_content), 4):
            header += "\t"
            for t in range(4):
                s = file_content[i + t]
                header += "{}, ".format(float.hex(s))
            header += "\n"
        header += "};\n\n"
        header += "#endif\n"
        fp.write(header)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'f:o:n:v:b:')
    file_path = output_file = size = var_name = None
    zero_vector = False
    for o, a in opts:
        if o == '-f':
            file_path = a

        if o == '-o':
            output_file = a

        if o == '-n':
            size = a

        if o == '-v':
            var_name = a

    if None in [file_path, output_file, size, var_name]:
        print("usage:")
        print("./convert_input.py -f <path to binary file> -o <output header> "
              "-n <number of float elements> -v <var name in header>")
        exit(0)

    generate_input_h(file_path=file_path, output_file=output_file, input_size=int(size), var_name=var_name)
