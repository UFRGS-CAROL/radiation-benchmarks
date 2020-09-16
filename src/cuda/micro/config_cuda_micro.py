#!/usr/bin/python3

import configparser
import sys

sys.path.insert(0, '../../include')

from common_config import discover_board, execute_and_write_json_to_file

ITERATIONS = int(1e9)
INT_MICRO = {
    "int32": {
        "branch": {"ops": 10000000, "block_size": 256, "sm_factor": 1}
    }

    # "float": {
    #     "fma": {"ops": 100000000, "block_size": 256, "sm_factor": 1}
    # }
}


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
                "make -j 4",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for precision, configs in INT_MICRO.items():
        for instruction, parameters in configs.items():
            block_size = parameters["block_size"]
            sm_factor = parameters["sm_factor"]
            ops = parameters["ops"]
            gold_path = data_path + f"/gold_{precision}_{instruction}_{block_size}_{sm_factor}_{ops}.data"
            gen = [
                ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                 bin_path + '/' + benchmark_bin + " "],
                [f'--iterations {ITERATIONS}'],
                [f'--gold {gold_path}'],
                [f'--precision {precision}'],
                [f'--inst {instruction}'],
                [f'--opnum {ops}'],
                [f'--blocksize {block_size}'],
                [f'--smmul {sm_factor}'],
            ]

            execute.append(' '.join(str(r) for v in gen for r in v))
            gen.append(['--generate'])
            generate.append(' '.join(str(r) for v in gen for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[-1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        pass

    board, hostname = discover_board()
    config(board=board, debug=debug_mode)
