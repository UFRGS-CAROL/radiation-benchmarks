import copy


ITERATIONS = int(1e9)
PRECISIONS = ["single"]

FAST_MATH = [0, 1]
OPS_LIST = [1, 8]
BUILD_PROFILER = 0

COMMON_MICROS_LIST = ["add"] #, "mul", "mad"]
INT_MICRO_LIST = [] #"branch", "ldst"]
FLOAT_MICRO_LIST = [] #"div", "fma", "euler", "pythagorean"]

STDDICT = {
    "fast_math": FAST_MATH,
    "ops_list": OPS_LIST,
    "precisions": PRECISIONS
}

COMMON_MICROS = {i: STDDICT for i in COMMON_MICROS_LIST}

INT_MICRO = {i: STDDICT for i in COMMON_MICROS_LIST}

for n in INT_MICRO_LIST:
    new_one = copy.deepcopy(STDDICT)
    new_one["fast_math"] = [0]
    new_one["ops_list"] = [0]
    INT_MICRO[n] = new_one


FLOAT_MICRO = copy.deepcopy(COMMON_MICROS)
if "mad" in FLOAT_MICRO:
    del FLOAT_MICRO["mad"]

for n in FLOAT_MICRO_LIST:
    new_one = copy.deepcopy(STDDICT)
    FLOAT_MICRO[n] = new_one

# Going from 128 instructions (minimum unroll)
# to 1k, 16MB, 512MB
# https://ieeexplore.ieee.org/abstract/document/5306801
