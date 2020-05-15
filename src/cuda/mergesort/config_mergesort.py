#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [134217728]
ITERATIONS = int(1e9)
BUILDPROFILER = 0


def config(board, debug):
    DATA_PATH_BASE = "mergesort"

    benchmark_bin = "mergesort"
    print("Generating " + benchmark_bin + " for CUDA, board:" + str(board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise ValueError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/" + DATA_PATH_BASE
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/mergesort"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0o777)
        os.chmod(data_path, 0o777)

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -C ../common/",
                "make -j2 BUILDPROFILER={}".format(BUILDPROFILER),
                "mkdir -p " + data_path,
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []
    for size in SIZES:
        input_file = data_path + "/"

        gen = [None] * 8
        gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                  bin_path + "/" + benchmark_bin + " "]

        gen[1] = ['-size=' + str(size)]
        gen[2] = ['-input=' + input_file + 'input_' + str(size)]
        gen[3] = ['-gold=' + input_file + 'gold_' + str(size)]
        gen[4] = ['-iterations=1']
        gen[5] = ['-noinputensurance']
        gen[6] = ['-verbose']
        gen[7] = ['-generate']

        # change mode and iterations for exe
        exe = copy.deepcopy(gen)
        exe[4] = ['-iterations={}'.format(ITERATIONS)]
        del exe[-2:]
        generate.append(' '.join(str(r) for v in gen for r in v))
        execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[0:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        debug_mode = False

    board, hostname = discover_board()
    config(board=hostname, debug=debug_mode)
