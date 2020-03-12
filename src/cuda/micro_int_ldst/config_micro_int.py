#!/usr/bin/python3
import configparser
import sys

sys.path.insert(0, '../../include')
sys.path.insert(0, '../micro')

from common_config import discover_board, execute_and_write_json_to_file
import common_micro_config as cm

# ITERATIONS = 2147483647
# OPERATIONS = 1000000
# MICROBENCHMARKS = ["ldst", "add", "mad", "mul"]
# BUILDPROFILER = 1


def config(board, debug):
    benchmark_bin = "cudaMicroIntLDST"
    print("Generating {} for CUDA, board: {}".format(benchmark_bin, board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = configparser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise ValueError("Configuration setup error: " + str(e))

    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/micro_int_ldst"
    data_path = install_dir + "data/micro"

    generate = ["sudo mkdir -p " + bin_path,
                "sudo mkdir -p " + data_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make -C ../common ",
                "make BUILDPROFILER={} LOGS=1".format(cm.BUILDPROFILER),
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]

    execute = []
    for inst_type in cm.INT_MICRO:
        print(inst_type)
        configs = cm.INT_MICRO[inst_type]
        ops_list = configs["ops_list"]
        for ops in ops_list:
            gold_path = data_path + "/gold_{}_{}.data".format(inst_type, ops)
            input_path = data_path + "/" + cm.GENERAL_INPUT
            gen = [
                ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                 bin_path + '/' + benchmark_bin + " "],
                ['--iterations {}'.format(cm.ITERATIONS)],
                ['--input {}'.format(input_path)],
                ['--gold {}'.format(gold_path)],
                ['--inst {}'.format(inst_type)],
                ['--opnum {}'.format(ops)],
                ['--generate']
            ]

            generate.append(' '.join(str(r) for v in gen for r in v))
            del gen[-1]
            execute.append(' '.join(str(r) for v in gen for r in v))

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
