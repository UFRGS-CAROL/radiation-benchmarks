#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [512, 1024]


def config(board, debug):
    DATA_PATH_BASE = "mxm"

    benchmark_bin = "matmul"
    print "Generating " + benchmark_bin + " For board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/sequential/mxm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    arch = 'x86'
    if 'raspberry' in board:
        arch = 'arm64'
    elif 'zedboard' in board:
        arch = 'arm'

    generate = [" mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include STATIC=1",
                "make ARCH={} LOGS=1 ".format(arch),
                "mkdir -p " + data_path,
                " rm -f " + data_path + "/*" + benchmark_bin + "*",
                " mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    for i in SIZES:
        input_file = "{}/matmul_input_{}.txt".format(data_path, i)
        gold_file = "{}/matmul_gold_{}.txt".format(data_path, i)

        # ./ input_matmul.py. / matmul_input.dat     1024
        generate.append('./input_matmul.py  {} {}'.format(input_file, i))

        # generate
        # ./ matmul. / matmul_input.dat. / matmul_gold.txt 1 1024
        gen = [None] * 5
        gen[0] = [bin_path + "/" + benchmark_bin + " "]
        gen[1] = [input_file]
        gen[2] = [gold_file]
        gen[3] = [1]  # change for execute
        gen[4] = [i]

        # change mode and iterations for exe
        # ./ matmul. / matmul_input.dat. / matmul_gold.txt 0 1024
        exe = copy.deepcopy(gen)
        exe[3] = [0]

        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
    print "Multiple jsons may have been generated."
