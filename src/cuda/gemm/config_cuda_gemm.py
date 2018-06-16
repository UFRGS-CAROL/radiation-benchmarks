#!/usr/bin/python

import ConfigParser
import copy
import os
import sys

sys.path.insert(0, '../../include')
from common_config import discover_board, execute_and_write_json_to_file

SIZES = [16384] #[8192, 2048, 512]
PRECISIONS = ["double", "single", "half"]
ITERATIONS = 100000
USE_TENSOR_CORES = [0, 1]
CHECK_INPUTS = [1] #[0, 1]

def config(board, arith_type, debug):

    DATA_PATH_BASE = "gemm_" + arith_type

    benchmark_bin = "cuda_gemm_" + arith_type
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
    src_benchmark = install_dir + "src/cuda/gemm"

    if not os.path.isdir(data_path):
        os.mkdir(data_path, 0777)
        os.chmod(data_path, 0777)

    max_size = max(SIZES)

    generate = ["sudo mkdir -p " + bin_path, 
                "cd " + src_benchmark, 
                "make clean", 
                "make -C ../../include ", 
                "make PRECISION=" + arith_type + " -j 4 " + "DEFAULT_INPUT_SIZE=" + max_size,
                "mkdir -p " + data_path, 
                "sudo rm -f " + data_path + "/*" + benchmark_bin + "*",
                "sudo mv -f ./" + benchmark_bin + " " + bin_path + "/"]
    execute = []

    for i in SIZES:
        for tc in USE_TENSOR_CORES:
            if  (arith_type == 'double') and ((tc == 1) and (0 in USE_TENSOR_CORES)):
                continue # TENSOR not implemented on Double
            for input_check in CHECK_INPUTS:
                input_file = data_path + "/"

                gen = [None] * 10
                gen[0] = ['sudo env LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} ', bin_path + "/" + benchmark_bin + " "]
                gen[1] = ['-size=' + str(i)]
                gen[2] = ['-input_a=' + input_file + 'A_' + str(arith_type) + "_" + str(max_size) + "_use_tensor_" + str(tc) + '.matrix']
                gen[3] = ['-input_b=' + input_file + 'B_' + str(arith_type) + "_" + str(max_size) + "_use_tensor_" + str(tc) + '.matrix']
                gen[4] = ['-gold=' + input_file + "GOLD_" + str(arith_type) + "_" +  str(i) + "_use_tensor_" + str(tc) + ".matrix"]  # change for execute
                gen[5] = []
                gen[6] = []
                gen[7] = ['-use_tensor=' + str(tc)]
                gen[8] = ['-input_check=' + str(input_check)]
                gen[9] = ['-generate']

                # change mode and iterations for exe
                exe = copy.deepcopy(gen)
                exe[0][1] = bin_path + '/' + benchmark_bin + " "
                exe[5] = ['-iterations=' + str(ITERATIONS)]
                exe[6] = ['-gpu_check']
                exe[8] = []

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
