#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

PRECISIONS = ['double', 'float']
SIZES = [[1024, 8], [2048, 16], [4096, 32]]
THREADS = [1, 4, 6]
ITERATIONS = 100000


def config(board, precision, debug):
    DATA_PATH_BASE = "gemm"

    benchmark_bin = "gemm_{}_check".format(precision)
    print("Generating {0} for OPENMP, board:{1}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print("Configuration setup error: {0}".format(str(e)))
        sys.exit(1)

    # arch define
    architecture = 'x86'
    if 'x2' in board or 'raspberry' in board:
        architecture = 'arm'
    elif 'xeon' in board:
        architecture = 'xeon'

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/openmp/gemm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    max_size = max(SIZES)[0]
    matrix_a = data_path + "/{}_gemm_openmp_input_matrix_a_{}".format(precision, max_size)
    matrix_b = data_path + "/{}_gemm_openmp_input_matrix_b_{}".format(precision, max_size)

    generate = ["mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make ARCH={} -j 4 LOGS=1 DEFAULT_INPUT={} PRECISION={}".format(architecture, max_size, precision),
                "mkdir -p " + data_path,
                "rm -f {} {}".format(matrix_a, matrix_b),
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]

    execute = []

    for [size, tiling] in SIZES:
        gen = [None] * 7
        gen[0] = ['{}/gemm_{}_gen'.format(src_benchmark, precision)]
        gen[1] = [4]
        gen[2] = [size]
        gen[3] = [tiling]
        gen[4] = [matrix_a]
        gen[5] = [matrix_b]
        gen[6] = [data_path + "/gold_openmp_{}_gemm_size_{}_tiling_{}".format(precision, size, tiling)]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[0] = ['{}/gemm_{}_check'.format(bin_path, precision)]
        exe.append([ITERATIONS])

        generate.append(' '.join(str(r) for v in gen for r in v))

        for thread in THREADS:
            exe[1] = [thread]
            execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[0:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False

    board, _ = discover_board()
    for p in PRECISIONS:
        config(board=board, precision=p, debug=debug_mode)
    print("Multiple jsons may have been generated.")
