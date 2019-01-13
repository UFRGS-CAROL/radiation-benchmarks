#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [8192]
THREADS = [1, 4, 6]
ITERATIONS = 100000


def config(board, debug):
    DATA_PATH_BASE = "dgemm"

    benchmark_bin = "openmp_gemm"
    print "Generating " + benchmark_bin + " for OPENMP, board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    # arch define
    architecture = 'x86'
    if 'x2' in board or 'raspberry' in board:
        architecture = 'arm'
    elif 'xeon' in board:
        architecture = 'xeon'

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/gemm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    max_size = max(SIZES)

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make ARCH=" + architecture + " -j 4 LOGS=1 DEFAULT_INPUT=8192",
                "mkdir -p " + data_path,
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]

    execute = []

    for i in SIZES:
        input_file = data_path + "/"

        gen = [None] * 10
        gen[0] = ['sudo ', bin_path + "/" + benchmark_bin + " "]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[9] = []

        generate.append(' '.join(str(r) for v in gen for r in v))
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
    config(board=board, debug=debug_mode)
    print "Multiple jsons may have been generated."
