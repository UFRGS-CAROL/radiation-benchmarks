#!/usr/bin/python

import ConfigParser
import copy
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = int(1e10)
PRECISIONS = ["double"]
DMR = ["dmr", "dmrmixed", "none"]
TYPES = ["fma", "add", "mul"]
CHECK_BLOCK = [1, 10, 100, 1000]


def config(board, debug):
    benchmark_bin = "cuda_micro_mp_hardening"
    print("Generating {} for CUDA, board: {}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise ValueError("Configuration setup error: " + str(e))

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/micro_mp_hard"
    data_path = install_dir + "data/micro_mp_hard"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + data_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for precision in PRECISIONS:
        for type_ in TYPES:
            gold_path = data_path + "/gold_{}_{}.data".format(type_, precision)
            gen = [None] * 8

            for dmr in DMR:
                    for op in CHECK_BLOCK:
                        gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                                  bin_path + '/' + benchmark_bin + " "]
                        gen[1] = ['--iterations {}'.format(ITERATIONS)]
                        gen[2] = ['--redundancy {}'.format(dmr)]
                        gen[3] = ['--inst {}'.format(type_)]
                        gen[4] = ['--gold {}'.format(gold_path)]
                        gen[5] = ['--opnum {}'.format(op)]
                        gen[6] = ['--precision {}'.format(precision)]
                        gen[7] = ['--generate']

                        exe = copy.deepcopy(gen)
                        exe[-1] = []
                        execute.append(' '.join(str(r) for v in exe for r in v))

            generate.append(' '.join(str(r) for v in gen for r in v))

    if debug is False:
        execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)
    else:
        for g in generate:
            print(g)
        print()
        for e in execute:
            print(e)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][0]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        pass

    board, _ = discover_board()
    config(board=board, debug=debug_mode)
