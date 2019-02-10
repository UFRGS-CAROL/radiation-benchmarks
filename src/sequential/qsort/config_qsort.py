#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [1024]


def config(board, debug):
    DATA_PATH_BASE = "qsort"

    benchmark_bin = "qsort"
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
    src_benchmark = install_dir + "src/sequential/qsort"

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

    for i in SIZES:
        input_file = "{}/quicksort_input_{}.txt".format(data_path, i)
        gold_file = "{}/quicksort_gold_{}.txt".format(data_path, i)

        # generate
        # ./qsort 2000000 0 ./quicksort_gold_2000000.txt ./quicksort_input_2000000.txt
        gen = [None] * 5
        gen[0] = ["sudo " + bin_path + "/" + benchmark_bin + " "]
        gen[1] = [i]
        gen[2] = [1]
        gen[3] = [gold_file]  # change for execute
        gen[4] = [input_file]

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[2] = [1]

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
