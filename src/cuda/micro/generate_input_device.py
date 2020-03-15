#!/usr/bin/python

import random
import struct

MAXBLOCK = 1024
NUMBERS_PER_LINE = 4
NUMBERS_PER_LINE_INT = 8

RANGE = 10.0


def float_to_hex(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]


with open("src/input_device.h", "w") as fp:
    fp.write("#ifndef INPUT_DEVICE_H_\n")
    fp.write("#define INPUT_DEVICE_H_\n\n")

    fp.write("__device__  __constant__ float common_float_input[MAX_THREAD_BLOCK] = {\n\t")
    random_list = [float(random.uniform(-RANGE, RANGE)) for _ in range(0, MAXBLOCK)]

    for i in range(0, MAXBLOCK, NUMBERS_PER_LINE):
        for number in range(0, NUMBERS_PER_LINE):
            random_float = random_list[i + number]
            fp.write("{}, ".format(random_float))
        fp.write("\n\t")

    fp.write("};\n\n")

    fp.write("__device__ __constant__ __restrict__ int32_t common_int_input[MAX_THREAD_BLOCK] = {\n\t")

    for i in range(0, MAXBLOCK, NUMBERS_PER_LINE_INT):
        for number in range(0, NUMBERS_PER_LINE_INT):
            random_float = float_to_hex(random_list[i + number])
            fp.write("{}, ".format(random_float & 0xFFFF))
        fp.write("\n\t")

    fp.write("};\n\n")

    fp.write("__device__  __constant__ __restrict__ int32_t inverse_mul_input[MAX_THREAD_BLOCK] = {\n\t")

    for i in range(0, MAXBLOCK, NUMBERS_PER_LINE_INT):
        for number in range(0, NUMBERS_PER_LINE_INT):
            random_float = float_to_hex(random_list[i + number]) & 0xFFFF
            random_float = 0x100000000 / random_float + 1
            fp.write("{}, ".format(random_float))
        fp.write("\n\t")

    fp.write("};\n\n")

    fp.write("#endif /* INPUT_DEVICE_H_ */\n")
