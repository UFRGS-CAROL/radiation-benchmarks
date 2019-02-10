#!/usr/bin/python

import ConfigParser
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

MODES = ['E', 'D']


def config(board, debug):
    DATA_PATH_BASE = "aes"

    benchmark_bin = "rijndael"
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
                " mv -f ./" + benchmark_bin + " " + bin_path + "/",
                " tar xzf {}/input.tar.gz -C {}/".format(data_path, data_path)]
    execute = []

    for i in MODES:
        exe = [None] * 6
        exe[0] = ["sudo " + bin_path + "/" + benchmark_bin + " "]
        exe[1] = ["127.0.0.1", "999"]
        if i == 'E':
            exe[2] = ["{}/input_large.asc".format(data_path)]
            exe[4] = ["{}/input_large.enc".format(data_path)]
        elif i == 'D':
            exe[2] = ["{}/input_large.enc".format(data_path)]
            exe[4] = ["{}/input_large.asc".format(data_path)]

        exe[3] = ["/home/fernando/git_pesquisa/radiation-benchmarks/data/aes/tmp.file"]
        exe[5] = [i, "1234567890abcdeffedcba09876543211234567890abcdeffedcba0987654321"]

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
