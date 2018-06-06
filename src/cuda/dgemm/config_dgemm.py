#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [8192, 2048, 4096]
ITERATIONS = 10000
USE_TENSOR_CORES = [0, 1]

DEBUG_MODE = False
BENCHMARK_BIN = "cudaDGEMM"
DATA_PATH_BASE = "dgemm"
GENERATE_BIN_NAME = "generateMatricesDouble"


def config(board, debug):
    benchmark_bin = BENCHMARK_BIN
    print "Generating " + benchmark_bin + " for CUDA, board:" + board

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
    src_dgemm = install_dir + "src/cuda/" + DATA_PATH_BASE

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    # change it for lava
    generate = ["sudo mkdir -p " + bin_path, "cd " + src_dgemm, "make clean", "make -C ../../include ", "make -j 4",
                "mkdir -p " + data_path, "sudo rm -f " + data_path + "/*",
                "mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size
    max_size = max(SIZES)
    for i in SIZES:
        for tc in USE_TENSOR_CORES:
            input_file = data_path + "/"

            gen = [None] * 7
            gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', src_dgemm + "/" + GENERATE_BIN_NAME + " "]
            gen[1] = ['-size=' + str(i)]
            gen[2] = ['-input_a=' + input_file + benchmark_bin + 'A_' + str(max_size) + '.matrix']
            gen[3] = ['-input_b=' + input_file + benchmark_bin + 'B_' + str(max_size) + '.matrix']
            gen[4] = ['-gold=' + input_file + "GOLD_" + str(i) + ".matrix"]  # change for execute
            gen[5] = []
            gen[6] = ['-use_tensors=' + str(tc)]

            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe[0][1] = bin_path + '/' + benchmark_bin + " "
            exe[5] = ['-iterations=' + str(ITERATIONS)]

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)



if __name__ == "__main__":
    try:
        parameter = str(sys.argv[1:][0]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    config(board=board, debug=debug_mode)
