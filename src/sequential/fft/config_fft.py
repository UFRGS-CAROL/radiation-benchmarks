#!/usr/bin/python

import ConfigParser
import os
import sys

import copy

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

# test
# ./fft 127.0.0.1 999 ./input.txt ./gold.txt 262144 8 0

MODES = [[262144, 8]]


def config(board, debug):
    benchmark_bin = "fft"
    print "Generating " + benchmark_bin + " For board:" + board

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        print >> sys.stderr, "Configuration setup error: " + str(e)
        sys.exit(1)

    data_path = install_dir + "data/" + benchmark_bin
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/sequential/" + benchmark_bin

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
                " mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for size, wave in MODES:
        # generate
        # ./fft 127.0.0.1 999 ./input.txt ./gold.txt 262144 8 1

        gen = [None] * 5
        gen[0] = [bin_path + "/" + benchmark_bin + " "]
        gen[1] = ["127.0.0.1", "999"]
        gen[2] = [data_path + "/fft_input_{}_{}".format(size, wave)]
        gen[3] = [data_path + "/fft_gold_{}_{}".format(size, wave)]
        gen[4] = [size, wave, 1]

        exe = copy.deepcopy(gen)
        exe[4][2] = 0

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
