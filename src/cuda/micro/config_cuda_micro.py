#!/usr/bin/python3

import configparser
import sys

sys.path.insert(0, '../../include')

from common_config import discover_board, execute_and_write_json_to_file
import common_micro_config as cm


def config(board, debug):
    benchmark_bin = "cudaMicro"
    print("Generating " + benchmark_bin + " for CUDA, board:" + str(board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
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
                "make BUILDPROFILER={} PRECISION=".format(cm.BUILD_PROFILER) + " -j 4",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for inst_type in cm.FLOAT_MICRO:
        print(inst_type)
        configs = cm.FLOAT_MICRO[inst_type]
        fast_math_list = configs["fast_math"]
        ops_list = configs["ops_list"]
        precisions = configs["precisions"]
        for precision in precisions:
            for ops in ops_list:
                for fast_math in fast_math_list:
                    gold_path = data_path + "/gold_{}_{}_{}_{}.data".format(ops, fast_math, inst_type, precision)
                    gen = [
                        ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                         bin_path + '/' + benchmark_bin + " "],
                        ['--iterations {}'.format(cm.ITERATIONS)],
                        ['--gold {}'.format(gold_path)],
                        ['--precision {}'.format(precision)],
                        ['--inst {}'.format(inst_type)],
                        ['--opnum {}'.format(ops)],
                    ]
                    if fast_math == 1:
                        gen.append(['--fast-math'])
                    gen.append(['--generate'])

                    generate.append(' '.join(str(r) for v in gen for r in v))
                    del gen[-1]
                    execute.append(' '.join(str(r) for v in gen for r in v))

    for inst_type in cm.INT_MICRO:
        print(inst_type)
        configs = cm.INT_MICRO[inst_type]
        ops_list = configs["ops_list"]
        for ops in ops_list:
            gold_path = data_path + "/gold_{}_{}_int32.data".format(inst_type, ops)
            gen = [
                ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                 bin_path + '/' + benchmark_bin + " "],
                ['--iterations {}'.format(cm.ITERATIONS)],
                ['--gold {}'.format(gold_path)],
                ['--precision int32'],
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
