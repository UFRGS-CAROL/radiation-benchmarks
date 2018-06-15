#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [4096, 2048, 512]
PRECISIONS = ["double", "single", "half"]
ITERATIONS = 10000
USE_TENSOR_CORES = [0] #, 1]


def config(board, arith_type, debug):

    DATA_PATH_BASE = "mxm_" + arith_type

    benchmark_bin = "cuda_trip_mxm_" + arith_type
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
    src_benchmark = install_dir + "src/cuda/trip_mxm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    generate = ["sudo mkdir -p " + bin_path, 
                "cd " + src_benchmark, 
                "make clean", 
                "make -C ../../include ", 
                "make PRECISION=" + arith_type + " -j 4",
                "mkdir -p " + data_path, 
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    # gen only for max size, defined on cuda_trip_mxm.cu
    max_size = 8192
    for i in SIZES:
        for tc in USE_TENSOR_CORES:
            input_file = data_path + "/"

            gen = [None] * 8
            gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', bin_path + "/" + benchmark_bin + " "]
            gen[1] = ['-size=' + str(i)]
            gen[2] = ['-input_a=' + input_file + 'A_' + str(max_size) + "_use_tensor_" + str(tc) + '.matrix']
            gen[3] = ['-input_b=' + input_file + 'B_' + str(max_size) + "_use_tensor_" + str(tc) + '.matrix']
            gen[4] = ['-gold=' + input_file + "GOLD_" +  str(i) + "_use_tensor_" + str(tc) + ".matrix"]  # change for execute
            gen[5] = []
            gen[6] = ['-use_tensors=' + str(tc)]
            gen[7] = ['-generate']

            # change mode and iterations for exe
            exe = copy.deepcopy(gen)
            exe[0][1] = bin_path + '/' + benchmark_bin + " "
            exe[5] = ['-iterations=' + str(ITERATIONS)]
            exe[7] = []

            generate.append(' '.join(str(r) for v in gen for r in v))
            execute.append(' '.join(str(r) for v in exe for r in v))

    execute_and_write_json_to_file(execute, generate, install_dir, benchmark_bin, debug=debug)



if __name__ == "__main__":
    try:
        parameter = str(sys.argv[1:][1]).upper() 
        if parameter == 'DEBUG':
            debug_mode = True
    except:
        debug_mode = False
    
    board, _ = discover_board()
    for p in PRECISIONS:
        config(board=board, arith_type=p, debug=debug_mode)
    print "Multiple jsons may have been generated."
