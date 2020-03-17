#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [8192, 1024]  # 4096
PRECISIONS = ["float"]  # , "half"]
ITERATIONS = 10000
USE_TENSOR_CORES = [0]
USE_CUBLAS = [0, 1]


def config(board, arith_type, debug):
    benchmark_bin = "gemm"
    print("Generating " + benchmark_bin + " for CUDA, board:" + str(board))

    conf_file = '/etc/radiation-benchmarks.conf'
    try:
        config = ConfigParser.RawConfigParser()
        config.read(conf_file)
        install_dir = config.get('DEFAULT', 'installdir') + "/"

    except IOError as e:
        raise IOError("Configuration setup error: " + str(e))

    data_path = install_dir + "data/gemm"
    bin_path = install_dir + "bin"
    src_benchmark = install_dir + "src/cuda/gemm_tensorcores"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0o777)
        os.chmod(data_path, 0o777)

    generate = ["sudo mkdir -p " + bin_path,
                "cd " + src_benchmark,
                "make clean",
                "make -C ../../include ",
                "make PRECISION=" + arith_type + " -j 4 LOGS=1",
                "mkdir -p " + data_path,
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    # max_size = max(SIZES)
    for i in SIZES:
        for tc in USE_TENSOR_CORES:
            for cublas in USE_CUBLAS:
                input_file = data_path + "/"

                gen = [
                    ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ',
                     bin_path + "/" + benchmark_bin + " "],
                    ['--size ' + str(i)],
                    ['--input_a ' + input_file + 'A_' + str(i) + "_use_tensor_" + str(tc) + '.matrix'],
                    ['--input_b ' + input_file + 'B_' + str(i) + "_use_tensor_" + str(tc) + '.matrix'],
                    ['--input_c ' + input_file + 'C_' + str(i) + "_use_tensor_" + str(tc) + '.matrix'],
                    ['--gold ' + input_file + "GOLD_" + str(i) + "_use_tensor_" + str(tc) + ".matrix"],
                    ['--verbose 0'],
                    ['--tensor_cores ' + str(tc)],
                    ['--triplicated 1'],
                    ['--precision ' + str(arith_type)],
                    ['--use_cublas' if cublas == 1 else ''],
                    ['--iterations ' + str(ITERATIONS)],
                    ['--generate 1']

                ]

                # change mode and iterations for exe
                exe = copy.deepcopy(gen)
                exe[-1] = ['--generate 0']

                generate.append(' '.join(str(r) for v in gen for r in v))
                execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)
    os.rename(install_dir + "/scripts/json_files/gemm_wmma.json",
              install_dir + "/scripts/json_files/gemm_wmma_" + arith_type + ".json")


if __name__ == "__main__":
    debug_mode = False
    try:
        parameter = str(sys.argv[1:][1]).upper()
        if parameter == 'DEBUG':
            debug_mode = True
    except IndexError:
        debug_mode = False

    board, _ = discover_board()
    for p in PRECISIONS:
        config(board=board, arith_type=p, debug=debug_mode)
