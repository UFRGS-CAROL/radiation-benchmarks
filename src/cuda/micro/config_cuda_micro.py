#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
s
from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = int(1e9)
PRECISIONS = ["single"]
TYPES = ["fma", "add", "mul", "pythagorean", "euler"]
FASTMATH = 1
OPS = {x: 10000000 for x in TYPES if x not in ["pythagorean", "euler"]}
OPS.update({"pythagorean": 50000, "euler": 40000000})
BUILDPROFILER = 0


def config(board, debug):
    benchmark_bin = "cudaMicro"
    print("Generating " + benchmark_bin + " for CUDA, board:" + str(board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/micro"
    data_path = install_dir + "data/micro"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + data_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -C ../common",
                "make BUILDPROFILER={} PRECISION=".format(BUILDPROFILER) + " -j 4",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for inst_type in TYPES:
        for precision in PRECISIONS:
            ops = OPS[inst_type]
            gold_path = data_path + "/gold_{}_{}.data".format(inst_type, precision)
            gen = [
                ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                 bin_path + '/' + benchmark_bin + " "],
                ['--iterations {}'.format(ITERATIONS)],
                ['--gold {}'.format(gold_path)],
                ['--precision {}'.format(precision)],
                ['--inst {}'.format(inst_type)],
                ['--opnum {}'.format(ops)],
                ['--generate']
            ]

            generate.append(' '.join(str(r) for v in gen for r in v))
            del gen[-1]
            execute.append(' '.join(str(r) for v in gen for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        pass

    board, hostname = discover_board()
    config(board=board, debug=debug_mode)
